/** \unpacker for me0
 *  \author J. Lee - UoS
 */
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/GEMRawToDigi/plugins/ME0RawToDigiModule.h"

using namespace gem;

ME0RawToDigiModule::ME0RawToDigiModule(const edm::ParameterSet & pset)
{
  fed_token = consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("InputLabel") );  
  useDBEMap_ = pset.getParameter<bool>("useDBEMap");
  produces<ME0DigiCollection>(); 
}

void ME0RawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector")); 
  desc.add<bool>("useDBEMap", false); 
}

void ME0RawToDigiModule::doBeginRun_(edm::Run const& rp, edm::EventSetup const& iSetup)
{
  if (useDBEMap_){
    edm::ESHandle<ME0EMap> me0EMap;
    iSetup.get<ME0EMapRcd>().get(me0EMap);
    m_me0EMap = me0EMap.product();
    m_me0ROMap = m_me0EMap->convert();
  }
  else {
    // no eMap, using dummy
    m_me0EMap = new ME0EMap();
    m_me0ROMap = m_me0EMap->convertDummy();
  }
}

void ME0RawToDigiModule::produce(edm::StreamID, edm::Event & e, const edm::EventSetup & iSetup) const
{
  auto outME0Digis = std::make_unique<ME0DigiCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  e.getByToken( fed_token, fed_buffers );
  
  for (unsigned int id=FEDNumbering::MINME0FEDID; id<=FEDNumbering::MINME0FEDID; ++id){ 
    const FEDRawData& fedData = fed_buffers->FEDData(id);
    
    int nWords = fedData.size()/sizeof(uint64_t);
    //std::cout <<"ME0RawToDigiModule words "<< nWords<<std::endl;
    
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
      uint16_t amcId = amcData->BID();

      // Fill GEB
      for (unsigned short j = 0; j < amcData->GDcount(); ++j){
	auto gebData = std::make_unique<GEBdata>();
	gebData->setChamberHeader(*(++word));
	
	unsigned int m_nvb = gebData->Vwh() / 3; // number of VFAT2 blocks. Eventually add here sanity check
	uint16_t gebId = gebData->InputID();
	
	for (unsigned short k = 0; k < m_nvb; k++){
	  auto vfatData = std::make_unique<VFATdata>();
	  vfatData->read_fw(*(++word));
	  vfatData->read_sw(*(++word));
	  vfatData->read_tw(*(++word));
	  gebData->v_add(*vfatData);
	  
	  uint16_t bc=vfatData->BC();
	  //uint8_t ec=vfatData->EC();
	  uint8_t b1010=vfatData->b1010();
	  uint8_t b1100=vfatData->b1100();
	  uint8_t b1110=vfatData->b1110();
	  uint16_t ChipID=vfatData->ChipID();
	  //int slot=vfatData->SlotNumber(); 
	  uint16_t crc = vfatData->crc();
	  uint16_t crc_check = vfatData->checkCRC();
	  bool Quality = (b1010==10) && (b1100==12) && (b1110==14) && (crc==crc_check);

	  if (crc!=crc_check) edm::LogWarning("ME0RawToDigiModule") << "DIFFERENT CRC :"<<crc<<"   "<<crc_check;
	  if (!Quality) edm::LogWarning("ME0RawToDigiModule") << "Quality "<< Quality;
	  
	  uint32_t vfatId = (amcId << 17) | (gebId << 12) | ChipID;
	  //need to add gebId to DB
	  if (useDBEMap_) vfatId = ChipID;
	    
	  //check if ChipID exists.
	  ME0ROmap::eCoord ec;
	  ec.vfatId = vfatId;
	  ec.channelId = 1;
	  if (!m_me0ROMap->isValidChipID(ec)){
	    edm::LogWarning("ME0RawToDigiModule") << "InValid ChipID :"<<ec.vfatId;
	    continue;
	  }
	  
	  for (int chan = 0; chan < 128; ++chan) {
	    uint8_t chan0xf = 0;
	    if (chan < 64) chan0xf = ((vfatData->lsData() >> chan) & 0x1);
	    else chan0xf = ((vfatData->msData() >> (chan-64)) & 0x1);

	    // no hits
	    if(chan0xf==0) continue;

	    ec.channelId = chan;
	    ME0ROmap::dCoord dc = m_me0ROMap->hitPosition(ec);
	    int bx = bc-25;
	    ME0DetId me0Id(dc.me0DetId);
	    ME0Digi digi(dc.stripId,bx);

	    // std::cout <<"ME0RawToDigiModule vfatId "<<ec.vfatId
	    // 	      <<" me0DetId "<< me0Id
	    // 	      <<" chan "<< ec.channelId
	    // 	      <<" strip "<< dc.stripId
	    // 	      <<" bx "<< digi.bx()
	    // 	      <<std::endl;
	    
	    outME0Digis.get()->insertDigi(me0Id,digi);	    
	  }
	}
		  	
	gebData->setChamberTrailer(*(++word));
	amcData->g_add(*gebData);
      }
      
      amcData->setGEMeventTrailer(*(++word));
      amcData->setAMCTrailer(*(++word));
      amc13Event->addAMCpayload(*amcData);
    }
    
    amc13Event->setAMC13trailer(*(++word));
    amc13Event->setCDFTrailer(*(++word));
  }
    
  e.put(std::move(outME0Digis));
}
