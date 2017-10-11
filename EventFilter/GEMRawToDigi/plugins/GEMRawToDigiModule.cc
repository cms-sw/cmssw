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
  fed_token = consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("InputObjects") );  
  produces<GEMDigiCollection>("MuonGEMDigis"); 
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects", edm::InputTag("rawDataCollector")); 
}

void GEMRawToDigiModule::beginRun(const edm::Run &run, const edm::EventSetup& iSetup)
{
  edm::ESHandle<GEMEMap> gemEMap;
  iSetup.get<GEMEMapRcd>().get(gemEMap); 
  m_gemEMap = gemEMap.product();
  m_gemROMap = m_gemEMap->convertCS();

}

void GEMRawToDigiModule::produce( edm::Event & e, const edm::EventSetup& iSetup ){
  ///reverse mapping for unPacker

  auto outGEMDigis = std::make_unique<GEMDigiCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  e.getByToken( fed_token, fed_buffers );

  for (unsigned int id=FEDNumbering::MINGEMFEDID; id<=FEDNumbering::MAXGEMFEDID; ++id){ 
    const FEDRawData& fedData = fed_buffers->FEDData(id);
    
    int nWords = fedData.size()/sizeof(uint64_t);
    if (nWords==0) continue;

    const unsigned char * data = fedData.data();
    
    AMC13Event * amc13Event = new AMC13Event();
    
    const uint64_t* word = reinterpret_cast<const uint64_t* >(data);
    amc13Event->setCDFHeader(*word);
    amc13Event->setAMC13header(*(++word));
    
    // Readout out AMC headers
    for (unsigned short i = 0; i < amc13Event->nAMC(); ++i){
      amc13Event->addAMCheader(*(++word));
    }

    // Readout out AMC payloads
    for (unsigned short i = 0; i < amc13Event->nAMC(); ++i){
      AMCdata * amcData = new AMCdata();
      
      amcData->setAMCheader1(*(++word));      
      amcData->setAMCheader2(*(++word));
      amcData->setGEMeventHeader(*(++word));

      // Fill GEB
      for (unsigned short j = 0; j < amcData->GDcount(); ++j){
	GEBdata * gebData = new GEBdata();
	gebData->setChamberHeader(*(++word));
	int m_nvb = gebData->Vwh() / 3; // number of VFAT2 blocks. Eventually add here sanity check

	for (unsigned short k = 0; k < m_nvb; k++){
	  VFATdata * vfatData = new VFATdata();
	  vfatData->read_fw(*(++word));
	  vfatData->read_sw(*(++word));
	  vfatData->read_tw(*(++word));
	  gebData->v_add(*vfatData);

	  uint16_t bc=vfatData->BC();
	  uint8_t ec=vfatData->EC();
	  uint8_t b1010=vfatData->b1010();
	  uint8_t b1100=vfatData->b1100();
	  uint8_t b1110=vfatData->b1110();
	  uint16_t  ChipID=vfatData->ChipID();
	  //int slot=vfatData->SlotNumber(); 
	  uint16_t crc = vfatData->crc();
	  uint16_t crc_check = checkCRC(vfatData);
	  bool Quality = (b1010==10) && (b1100==12) && (b1110==14) && (crc==crc_check) ;
	  uint64_t converted=ChipID+0xf000;    

	  uint8_t chan0xf = 0;
	  
	  for (int chan = 0; chan < 128; ++chan) {
	    if (chan < 64){
	      chan0xf = ((vfatData->lsData() >> chan) & 0x1);
	    }
	    else {
	      chan0xf = ((vfatData->msData() >> (chan-64)) & 0x1);
	    }

	    if(chan0xf==0) continue;  

	    GEMROmap::eCoord ec;
	    ec.chamberId=31;
	    ec.vfatId = ChipID+0xf000;
	    ec.channelId = chan+1;
	    GEMROmap::dCoord dc = m_gemROMap->hitPosition(ec);

	    int strip=dc.stripId +1;//
	    if (strip > 2*128) strip-=128*2;
	    else if (strip < 128) strip+=128*2;

	    int etaP=dc.etaId;	  
	    // NEED TOO FIX GEMDETID
	    //GEMDetId gemId = getGEMDetID(ChipID);
	    
	    GEMDigi digi(strip,bc);
	    outGEMDigis.get()->insertDigi(GEMDetId(1,1,1,1,1,etaP),digi); 
	  }
	  
	  delete vfatData;
	}
	
	gebData->setChamberTrailer(*(++word));
	amcData->g_add(*gebData);
	delete gebData;
      }
      
      amcData->setGEMeventTrailer(*(++word));
      amcData->setAMCTrailer(*(++word));
      amc13Event->addAMCpayload(*amcData);
      delete amcData;
    }
    
    amc13Event->setAMC13trailer(*(++word));
    amc13Event->setCDFTrailer(*(++word));
  }
  
  e.put(std::move(outGEMDigis), "MuonGEMDigis");
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
  vfatBlockWords[1] = (0x000000000000ffff & vfatData->lsData());

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
