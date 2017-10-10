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

#include "CondFormats/DataRecord/interface/GEMChamberMapRcd.h"
#include "CondFormats/GEMObjects/interface/GEMChamberMap.h"
#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"
#include "EventFilter/GEMRawToDigi/plugins/GEMRawToDigiModule.h"

GEMRawToDigiModule::GEMRawToDigiModule(const edm::ParameterSet & pset)
{
  fed_token = consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("InputObjects") );  
  produces<GEMDigiCollection>("MuonGEMDigis"); 
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects", edm::InputTag("rawDataCollector")); 
}

void GEMRawToDigiModule::produce( edm::Event & e, const edm::EventSetup& c ){
  ///reverse mapping for unPacker
  edm::ESHandle<GEMChamberMap> gemChamberMap;
  c.get<GEMChamberMapRcd>().get(gemChamberMap); 
  const GEMChamberMap* theMapping = gemChamberMap.product();

  auto outGEMDigis = std::make_unique<GEMDigiCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  e.getByToken( fed_token, fed_buffers );

  for (unsigned int id=FEDNumbering::MINGEMFEDID; id<=FEDNumbering::MAXGEMFEDID; ++id){ 
    const FEDRawData& fedData = fed_buffers->FEDData(id);
    
    
    int nWords = fedData.size()/sizeof(uint64_t);
    if (nWords==0) continue;

    const unsigned char * data = fedData.data();
    
    m_AMC13Event = new AMC13Event();
    
    const uint64_t* header = reinterpret_cast<const uint64_t* >(data);
    m_AMC13Event->setCDFHeader(header);

    
    const uint64_t* word = (const uint64_t*)(header+1);
    m_AMC13Event->setAMC13header(word);
    
    // Readout out AMC headers
    for (unsigned short i = 0; i < m_AMC13Event->nAMC(); i++){
      m_AMC13Event->addAMCheader(++word);
    }

    // Readout out AMC payloads
    for (unsigned short i = 0; i < m_AMC13Event->nAMC(); i++){
      AMCdata * m_amcdata = new AMCdata();
      
      m_amcdata->setAMCheader1(++word);      
      m_amcdata->setAMCheader2(++word);
      m_amcdata->setGEMeventHeader(++word);

      // Fill GEB
      for (unsigned short j = 0; j < m_amcdata->GDcount(); j++){
	GEBdata * m_gebdata = new GEBdata();
	m_gebdata->setChamberHeader(++word);
	int m_nvb = m_gebdata->Vwh() / 3; // number of VFAT2 blocks. Eventually add here sanity check

	for (unsigned short k = 0; k < m_nvb; k++){
	  VFATdata * m_vfatdata = new VFATdata();
	  m_vfatdata->read_fw(++word);
	  m_vfatdata->read_sw(++word);
	  m_vfatdata->read_tw(++word);
	  m_gebdata->v_add(*m_vfatdata);


	  uint16_t bc=m_vfatdata->BC();
	  uint8_t ec=m_vfatdata->EC();
	  uint8_t b1010=m_vfatdata->b1010();
	  uint8_t b1100=m_vfatdata->b1100();
	  uint8_t b1110=m_vfatdata->b1110();
	  uint16_t  ChipID=m_vfatdata->ChipID();
	  //int slot=m_vfatdata->SlotNumber(); 
	  uint16_t crc = m_vfatdata->crc();
	  uint16_t crc_check = checkCRC(m_vfatdata);
	  bool Quality = (b1010==10) && (b1100==12) && (b1110==14) && (crc==crc_check) ;
	  uint64_t converted=ChipID+0xf000;    

	  for(int chan = 0; chan < 128; ++chan) {
	    if(chan < 64){
	      chan0xf = ((m_vfatdata->lsData() >> chan) & 0x1);
	    } else {
	      chan0xf = ((m_vfatdata->msData() >> (chan-64)) & 0x1);
	    }

	    if(chan0xf==0) continue;  

	    GEMROmap::eCoord ec;
	    ec.chamberId=31;
	    ec.vfatId = ChipID+0xf000;
	    ec.channelId = chan+1;
	    GEMROmap::dCoord dc = romapV2->hitPosition(ec);

	    int strip=dc.stripId +1;//
	    if (strip > 2*128) strip-=128*2;
	    else if (strip < 128) strip+=128*2;

	    int etaP=dc.etaId;	  

	    GEMDigi digi(strip,bc);
	    // bx is a single digi, where we should give
	    // NEED TOO FIX GEMDETID
	    // NEED TOO FIX GEMDETID
	    // NEED TOO FIX GEMDETID
	    outGEMDigis.get()->insertDigi(GEMDetId(1,1,1,chamberPosition,schamberPosition,etaP),digi); 
	  }
	  
	  delete m_vfatdata;
	}
	m_gebdata->setChamberTrailer(++word);
	m_amcdata->g_add(*m_gebdata);
	delete m_gebdata;
      }
      m_amcdata->setGEMeventTrailer(++word);
      m_amcdata->setAMCTrailer(++word);
      m_AMC13Event->addAMCpayload(*m_amcdata);
      delete m_amcdata;
    }
    m_AMC13Event->setAMC13trailer(++word);
    m_AMC13Event->setCDFTrailer(++word);
  }
  e.put(std::move(outGEMDigis), "MuonGEMDigis");
}


