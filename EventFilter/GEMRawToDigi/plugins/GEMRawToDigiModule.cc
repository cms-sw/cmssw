/** \unpacker for gem
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

#include "EventFilter/GEMRawToDigi/plugins/GEMRawToDigiModule.h"

using namespace gem;

GEMRawToDigiModule::GEMRawToDigiModule(const edm::ParameterSet & pset)
{
  fed_token = consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("InputLabel") );  
  useDBEMap_ = pset.getParameter<bool>("useDBEMap");
  produces<GEMDigiCollection>(); 
  unpackStatusDigis_ = pset.getParameter<bool>("UnpackStatusDigis");
  if (unpackStatusDigis_){
    produces<GEMVfatStatusDigiCollection>("vfatStatus"); 
    produces<GEMGEBStatusDigiCollection>("GEBStatus"); 
  }
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector")); 
  desc.add<bool>("useDBEMap", false); 
}

void GEMRawToDigiModule::doBeginRun_(edm::Run const& rp, edm::EventSetup const& iSetup)
{
  if (useDBEMap_){
    edm::ESHandle<GEMEMap> gemEMap;
    iSetup.get<GEMEMapRcd>().get(gemEMap);
    m_gemEMap = gemEMap.product();
    m_gemROMap = m_gemEMap->convert();
  }
  else {
    // no eMap, using dummy
    m_gemEMap = new GEMEMap();
    m_gemROMap = m_gemEMap->convertDummy();
  }
}

void GEMRawToDigiModule::produce(edm::StreamID, edm::Event & e, const edm::EventSetup & iSetup) const
{
  auto outGEMDigis = std::make_unique<GEMDigiCollection>();
  auto outVfatStatus = std::make_unique<GEMVfatStatusDigiCollection>();
  auto outGEBStatus = std::make_unique<GEMGEBStatusDigiCollection>();
  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  e.getByToken( fed_token, fed_buffers );
  
  for (unsigned int id=FEDNumbering::MINGEMFEDID; id<=FEDNumbering::MINGEMFEDID; ++id){ 
    const FEDRawData& fedData = fed_buffers->FEDData(id);
    
    int nWords = fedData.size()/sizeof(uint64_t);
    //std::cout <<"GEMRawToDigiModule words "<< nWords<<std::endl;
    
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
	GEMDetId gemId(-1,1,1,1,1,0); // temp ID
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

	  if (crc!=crc_check) edm::LogWarning("GEMRawToDigiModule") << "DIFFERENT CRC :"<<crc<<"   "<<crc_check;
	  if (!Quality) edm::LogWarning("GEMRawToDigiModule") << "Quality "<< Quality;
	  	  
	  uint32_t vfatId = (amcId << 17) | (gebId << 12) | ChipID;
	  //need to add gebId to DB
	  if (useDBEMap_) vfatId = ChipID;
	    
	  //check if ChipID exists.
	  GEMROmap::eCoord ec;
	  ec.vfatId = vfatId;
	  ec.channelId = 1;
	  if (!m_gemROMap->isValidChipID(ec)){
	    edm::LogWarning("GEMRawToDigiModule") << "InValid ChipID :"<<ec.vfatId;
	    continue;
	  }
	   
	  for (int chan = 0; chan < 128; ++chan) {
	    uint8_t chan0xf = 0;
	    if (chan < 64) chan0xf = ((vfatData->lsData() >> chan) & 0x1);
	    else chan0xf = ((vfatData->msData() >> (chan-64)) & 0x1);

	    // no hits
	    if(chan0xf==0) continue;

	    ec.channelId = chan;
	    GEMROmap::dCoord dc = m_gemROMap->hitPosition(ec);
	    int bx = bc-25;
	    gemId = dc.gemDetId;
	    GEMDigi digi(dc.stripId,bx);

	    // std::cout <<"GEMRawToDigiModule vfatId "<<ec.vfatId
	    // 	      <<" gemDetId "<< gemId
	    // 	      <<" chan "<< ec.channelId
	    // 	      <<" strip "<< dc.stripId
	    // 	      <<" bx "<< digi.bx()
	    // 	      <<std::endl;
	    
	    outGEMDigis.get()->insertDigi(gemId,digi);	    
	  }

          if (unpackStatusDigis_){
	    GEMVfatStatusDigi vfatStatus(b1010, b1100, vfatData->Flag(), b1110, vfatData->lsData(),
					 vfatData->msData(), crc, vfatData->crc_calc(), vfatData->isBlockGood());
            outVfatStatus.get()->insertDigi(gemId,vfatStatus);
	  }
	  
	}
	
	gebData->setChamberTrailer(*(++word));
        if (unpackStatusDigis_){
          GEMGEBStatusDigi gebStatus(gebData->ZeroSup(),
                                     gebData->InputID(),
                                     gebData->Vwh(),
                                     gebData->ErrorC(),
                                     gebData->OHCRC(),
                                     gebData->Vwt(),
                                     gebData->InFu(),
                                     gebData->Stuckd(),
                                     gebData->GEBflag());
          outGEBStatus.get()->insertDigi(gemId.chamberId(),gebStatus); 
        }
		  	
	amcData->g_add(*gebData);
      }
      
      amcData->setGEMeventTrailer(*(++word));
      amcData->setAMCTrailer(*(++word));
      amc13Event->addAMCpayload(*amcData);
    }
    
    amc13Event->setAMC13trailer(*(++word));
    amc13Event->setCDFTrailer(*(++word));
  }
  
  e.put(std::move(outGEMDigis));
  if (unpackStatusDigis_){
    e.put(std::move(outVfatStatus), "vfatStatus");
    e.put(std::move(outGEBStatus), "GEBStatus");
  }
}
