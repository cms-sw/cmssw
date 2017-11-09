/** \file
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
  unpackStatusDigis  = pset.getParameter<bool>("UnpackStatusDigis");
  if (unpackStatusDigis){
    produces<GEMVfatStatusDigiCollection>("vfatStatus"); 
    produces<GEMGEBStatusDigiCollection>("GEBStatus"); 
  }
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel", edm::InputTag("rawDataCollector")); 
  desc.add<bool>("useDBEMap", false); 
}

void GEMRawToDigiModule::beginRun(const edm::Run &run, const edm::EventSetup& iSetup)
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

void GEMRawToDigiModule::produce( edm::Event & e, const edm::EventSetup& iSetup )
{
  auto outGEMDigis = std::make_unique<GEMDigiCollection>();

  auto outVfatStatus = std::make_unique<GEMVfatStatusDigiCollection>();
  auto outGEBStatus = std::make_unique<GEMGEBStatusDigiCollection>();
  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  e.getByToken( fed_token, fed_buffers );

  int ndigis = 0;
  
  for (unsigned int id=FEDNumbering::MINGEMFEDID; id<=FEDNumbering::MINGEMFEDID; ++id){ 
    std::cout <<"GEMRawToDigiModule start "<<std::endl;
    const FEDRawData& fedData = fed_buffers->FEDData(id);
    
    int nWords = fedData.size()/sizeof(uint64_t);
    std::cout <<"GEMRawToDigiModule words "<< nWords<<std::endl;
    
    if (nWords<10) continue;
    const unsigned char * data = fedData.data();
    
    auto amc13Event = std::make_unique<AMC13Event>();
    
    const uint64_t* word = reinterpret_cast<const uint64_t* >(data);
    
    amc13Event->setCDFHeader(*word);
    amc13Event->setAMC13header(*(++word));

    // Readout out AMC payloads
    for (unsigned short i = 0; i < amc13Event->nAMC(); ++i){
      auto amcData = std::make_unique<AMCdata>();
      amcData->setAMCheader1(*(++word));      
      amcData->setAMCheader2(*(++word));
      amcData->setGEMeventHeader(*(++word));

      // Fill GEB
      for (unsigned short j = 0; j < amcData->GDcount(); ++j){
	auto gebData = std::make_unique<GEBdata>();
	gebData->setChamberHeader(*(++word));
	
	unsigned int m_nvb = gebData->Vwh() / 3; // number of VFAT2 blocks. Eventually add here sanity check
	int gebID = gebData->InputID();
	
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
	  uint16_t crc_check = checkCRC(vfatData.get());
	  bool Quality = (b1010==10) && (b1100==12) && (b1110==14) && (crc==crc_check);

	  if (crc!=crc_check) std::cout<<"DIFFERENT CRC :"<<crc<<"   "<<crc_check<<std::endl;
	  if (!Quality) std::cout <<"GEMRawToDigiModule Quality "<< Quality <<std::endl;
	  
	  uint16_t vfatId = ChipID | gebID << 12;
	  //need to add gebId to DB
	  if (useDBEMap_) vfatId = ChipID;
	    
	  //check if ChipID exists.
	  GEMROmap::eCoord ec;
	  ec.vfatId = vfatId;
	  ec.channelId = 1;	  
	  if (!m_gemROMap->isValidChipID(ec)){
	    std::cout <<"GEMRawToDigiModule InValid ChipID "<< ec.vfatId <<std::endl;	    
	    //delete vfatData;
	    continue;
	  }
          if (unpackStatusDigis){
	    GEMVfatStatusDigi vfatStatus(b1010, b1100, vfatData->Flag(), b1110, vfatData->lsData(), vfatData->msData(), crc, vfatData->crc_calc(), vfatData->isBlockGood());
            GEMROmap::dCoord dc = m_gemROMap->hitPosition(ec);
            GEMDetId tmpDetId(dc.gemDetId);
            outVfatStatus.get()->insertDigi(tmpDetId,vfatStatus);
	  }
	  for (int chan = 0; chan < 128; ++chan) {
	    uint8_t chan0xf = 0;
	    if (chan < 64) chan0xf = ((vfatData->lsData() >> chan) & 0x1);
	    else chan0xf = ((vfatData->msData() >> (chan-64)) & 0x1);

	    // no hits
	    if(chan0xf==0) continue;  

	    ec.channelId = chan;
	    GEMROmap::dCoord dc = m_gemROMap->hitPosition(ec);
	    
	    GEMDetId gemDetId(dc.gemDetId);
	    GEMDigi digi(dc.stripId,bc);

	    std::cout <<"GEMRawToDigiModule ChipID "<<ec.vfatId
		      <<" gemDetId "<< gemDetId
	    	      <<" chan "<< ec.channelId
	    	      <<" strip "<< dc.stripId
	    	      <<std::endl;
	    ndigis++;
	    
	    outGEMDigis.get()->insertDigi(gemDetId,digi);	    
	  }
	}
        if (unpackStatusDigis){
          GEMGEBStatusDigi gebStatus(gebData->ZeroSup(),
                                     gebData->InputID(),
                                     gebData->Vwh(),
                                     gebData->ErrorC(),
                                     gebData->OHCRC(),
                                     gebData->Vwt(),
                                     gebData->InFu(),
                                     gebData->Stuckd(),
                                     gebData->GEBflag());
          // need to update with GEB ID 
          //GEMDetId tmpDetId(dc.gemDetId.region(), dc.gemDetId.ring(), dc.gemDetId.station(), dc.gemDetId.layer(), dc.gemDetId.chamber(),0);
          GEMDetId tmpDetId(-1,1,1,1,1,0);
          outGEBStatus.get()->insertDigi(tmpDetId,gebStatus); 
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
  
  std::cout << "GEMRawToDigiModule ndigis " << ndigis << std::endl;
  
  e.put(std::move(outGEMDigis));
  if (unpackStatusDigis){
    e.put(std::move(outVfatStatus), "vfatStatus");
    e.put(std::move(outGEBStatus), "GEBStatus");
  }
}

uint16_t GEMRawToDigiModule::checkCRC(VFATdata * vfatData)
{
  uint16_t vfatBlockWords[12]; 
  vfatBlockWords[11] = ((0x000f & vfatData->b1010())<<12) | vfatData->BC();
  vfatBlockWords[10] = ((0x000f & vfatData->b1100())<<12) | ((0x00ff & vfatData->EC()) <<4) | (0x000f & vfatData->Flag());
  vfatBlockWords[9]  = ((0x000f & vfatData->b1110())<<12) | vfatData->ChipID();
  vfatBlockWords[8]  = (0xffff000000000000 & vfatData->msData()) >> 48;
  vfatBlockWords[7]  = (0x0000ffff00000000 & vfatData->msData()) >> 32;
  vfatBlockWords[6]  = (0x00000000ffff0000 & vfatData->msData()) >> 16;
  vfatBlockWords[5]  = (0x000000000000ffff & vfatData->msData());
  vfatBlockWords[4]  = (0xffff000000000000 & vfatData->lsData()) >> 48;
  vfatBlockWords[3]  = (0x0000ffff00000000 & vfatData->lsData()) >> 32;
  vfatBlockWords[2]  = (0x00000000ffff0000 & vfatData->lsData()) >> 16;
  vfatBlockWords[1]  = (0x000000000000ffff & vfatData->lsData());

  uint16_t crc_fin = 0xffff;
  for (int i = 11; i >= 1; i--){
    crc_fin = this->crc_cal(crc_fin, vfatBlockWords[i]);
  }
  
  return(crc_fin);
}

uint16_t GEMRawToDigiModule::crc_cal(uint16_t crc_in, uint16_t dato)
{
  uint16_t v = 0x0001;
  uint16_t mask = 0x0001;
  bool d=0;
  uint16_t crc_temp = crc_in;
  unsigned char datalen = 16;

  for (int i=0; i<datalen; i++){
    if (dato & v) d = 1;
    else d = 0;
    if ((crc_temp & mask)^d) crc_temp = crc_temp>>1 ^ 0x8408;
    else crc_temp = crc_temp>>1;
    v<<=1;
  }
  
  return(crc_temp);
}

