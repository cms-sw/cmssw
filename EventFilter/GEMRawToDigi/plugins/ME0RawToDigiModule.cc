/** \unpacker for me0
 *  \author J. Lee - UoS
 */
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/GEMRawToDigi/plugins/ME0RawToDigiModule.h"

using namespace gem;

ME0RawToDigiModule::ME0RawToDigiModule(const edm::ParameterSet & pset) :
  fed_token(consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("InputLabel") )),
  useDBEMap_(pset.getParameter<bool>("useDBEMap"))
{
  produces<ME0DigiCollection>(); 
}

void ME0RawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector")); 
  desc.add<bool>("useDBEMap", false);
  descriptions.add("muonME0Digis", desc);  
}

std::shared_ptr<ME0ROmap> ME0RawToDigiModule::globalBeginRun(edm::Run const&, edm::EventSetup const& iSetup) const
{
  auto me0ORmap = std::make_shared<ME0ROmap>();
  if (useDBEMap_){
    edm::ESHandle<ME0EMap> me0EMapRcd;
    iSetup.get<ME0EMapRcd>().get(me0EMapRcd);
    auto me0EMap = std::make_unique<ME0EMap>(*(me0EMapRcd.product()));
    me0EMap->convert(*me0ORmap);
    me0EMap.reset();    
  }
  else {
    // no EMap in DB, using dummy
    auto me0EMap = std::make_unique<ME0EMap>();
    me0EMap->convertDummy(*me0ORmap);
    me0EMap.reset();    
  }
  return me0ORmap;
}

void ME0RawToDigiModule::produce(edm::StreamID iID, edm::Event & iEvent, edm::EventSetup const&) const
{
  auto outME0Digis = std::make_unique<ME0DigiCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  iEvent.getByToken( fed_token, fed_buffers );
  
  auto me0ROMap = runCache(iEvent.getRun().index());
  
  for (unsigned int id=FEDNumbering::MINME0FEDID; id<=FEDNumbering::MINME0FEDID; ++id){ 
    const FEDRawData& fedData = fed_buffers->FEDData(id);
    
    int nWords = fedData.size()/sizeof(uint64_t);
    LogDebug("ME0RawToDigiModule") <<" words " << nWords;
    if (nWords<5) continue;
    const unsigned char * data = fedData.data();
    
    auto amc13Event = std::make_unique<AMC13Event>();
    
    const uint64_t* word = reinterpret_cast<const uint64_t* >(data);
    
    amc13Event->setCDFHeader(*word);
    amc13Event->setAMC13header(*(++word));

    // Readout out AMC headers
    for (unsigned short i = 0; i < amc13Event->nAMC(); ++i)
      amc13Event->addAMCheader(*(++word));
    
    // Readout out AMC payloads
    for (unsigned short i = 0; i < amc13Event->nAMC(); ++i){
      auto amcData = std::make_unique<AMCdata>();
      amcData->setAMCheader1(*(++word));      
      amcData->setAMCheader2(*(++word));
      amcData->setGEMeventHeader(*(++word));
      uint16_t amcId = amcData->boardId();
      uint16_t amcBx = amcData->bx();

      // Fill GEB
      for (unsigned short j = 0; j < amcData->gdCount(); ++j){
	auto gebData = std::make_unique<GEBdata>();
	gebData->setChamberHeader(*(++word));
	
	unsigned int m_nvb = gebData->vwh() / 3; // number of VFAT2 blocks
	uint16_t gebId = gebData->inputID();
	ME0DetId me0Id(-1,1,1,1); // temp ID
	for (unsigned short k = 0; k < m_nvb; k++){
	  auto vfatData = std::make_unique<VFATdata>();
	  vfatData->read_fw(*(++word));
	  vfatData->read_sw(*(++word));
	  vfatData->read_tw(*(++word));
	  gebData->addVFAT(*vfatData);
	  
	  uint16_t bc=vfatData->bc();
	  uint8_t b1010=vfatData->b1010();
	  uint8_t b1100=vfatData->b1100();
	  uint8_t b1110=vfatData->b1110();
	  uint16_t vfatId=vfatData->chipID();
	  uint16_t crc = vfatData->crc();
	  uint16_t crc_check = vfatData->checkCRC();
	  bool Quality = (b1010==10) && (b1100==12) && (b1110==14) && (crc==crc_check);

	  if (crc!=crc_check) edm::LogWarning("ME0RawToDigiModule") << "DIFFERENT CRC :"<<crc<<"   "<<crc_check;
	  if (!Quality) edm::LogWarning("ME0RawToDigiModule") << "Quality "<< Quality
							      << " b1010 "<< int(b1010)
							      << " b1100 "<< int(b1100)
							      << " b1110 "<< int(b1110);
	  
	  //check if ChipID exists.
	  ME0ROmap::eCoord ec = {amcId, gebId, vfatId};

	  if (!me0ROMap->isValidChipID(ec)){
	    edm::LogWarning("ME0RawToDigiModule") << "InValid ChipID :"<<ec.vfatId;
	    continue;
	  }
	  
	  for (int chan = 0; chan < VFATdata::nChannels; ++chan) {
	    uint8_t chan0xf = 0;
	    if (chan < 64) chan0xf = ((vfatData->lsData() >> chan) & 0x1);
	    else chan0xf = ((vfatData->msData() >> (chan-64)) & 0x1);

	    // no hits
	    if(chan0xf==0) continue;
	    ME0ROmap::dCoord dc = me0ROMap->hitPosition(ec);
	    // strip bx = vfat bx - amc bx
	    int bx = bc-amcBx;
	    me0Id = dc.me0DetId;
         
            ME0ROmap::channelNum chMap = {dc.vfatType, chan};
            ME0ROmap::stripNum stMap = me0ROMap->hitPosition(chMap);

            int stripId = stMap.stNum + (dc.iPhi-1)%ME0EMap::maxVFat_*ME0EMap::maxChan_;    

	    ME0Digi digi(stripId,bx);
	    LogDebug("ME0RawToDigiModule") <<" vfatId "<<ec.vfatId
					   <<" me0DetId "<< me0Id
					   <<" chan "<< chMap.chNum
					   <<" strip "<< stripId
					   <<" bx "<< digi.bx();

	    outME0Digis.get()->insertDigi(me0Id,digi);	    
	  }

	}
	
	gebData->setChamberTrailer(*(++word));
	amcData->addGEB(*gebData);
      }
      
      amcData->setGEMeventTrailer(*(++word));
      amcData->setAMCTrailer(*(++word));
      amc13Event->addAMCpayload(*amcData);
    }
    
    amc13Event->setAMC13trailer(*(++word));
    amc13Event->setCDFTrailer(*(++word));
  }
  
  iEvent.put(std::move(outME0Digis));
}
