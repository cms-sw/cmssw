/** \file
 *  \author J. Lee - UoS
 */

#include "EventFilter/GEMRawToDigi/plugins/GEMRawToDigiModule.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/DataRecord/interface/GEMChamberMapRcd.h"

GEMRawToDigiModule::GEMRawToDigiModule(const edm::ParameterSet & pset): 
  unPacker(new GEMRawToDigi(pset))
{
  fed_token = consumes<FEDRawDataCollection>( pset.getParameter<edm::InputTag>("InputObjects") );
  
  produces<GEMDigiCollection>("MuonGEMDigis"); 
  // produces<GEMPadDigiCollection>("MuonGEMPadDigis"); 
  // produces<GEMPadDigiClusterCollection>("MuonGEMPadDigiClusters"); 
  // produces<GEMCoPadDigiCollection>("CscTriggerPrimitiveDigis"); 
}

GEMRawToDigiModule::~GEMRawToDigiModule(){
  delete unPacker;
}

void GEMRawToDigiModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputObjects", edm::InputTag("rawDataCollector")); 
  descriptions.add("gemUnPacker", desc);
}

void GEMRawToDigiModule::produce( edm::Event & e, const edm::EventSetup& c ){
  ///reverse mapping for unPacker
  edm::ESHandle<GEMChamberMap> gemChamberMap;
  c.get<GEMChamberMapRcd>().get(gemChamberMap); 
  const GEMChamberMap* theMapping = gemChamberMap.product();

  auto gemDigi = std::make_unique<GEMDigiCollection>();
  // auto gemPadDigi = std::make_unique<GEMPadDigiCollection>();
  // auto gemPadDigiCluster = std::make_unique<GEMPadDigiClusterCollection>();
  // auto gemCoPadDigi = std::make_unique<GEMCoPadDigiCollection>();

  // Take raw from the event
  edm::Handle<FEDRawDataCollection> fed_buffers;
  e.getByToken( fed_token, fed_buffers );

  for (unsigned int id=FEDNumbering::MINGEMFEDID; id<=FEDNumbering::MAXGEMFEDID; ++id){ 
    const FEDRawData& feddata = fed_buffers->FEDData(id);
    
    if (feddata.size()){
      const unsigned char * data = feddata.data();
      uint64_t m_word;
      m_AMC13Event = new AMC13Event();
      m_AMC13Event->setCDFHeader(m_word);

      m_AMC13Event->setAMC13header(m_word);




      int bx=0;  
      uint8_t chan0xf = 0;

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

	GEMDigi digi(strip,bx); 
	// bx is a single digi, where we should give 
	// in input the strip and bx relative to trigger
	gemDigi.get()->insertDigi(GEMDetId(region,ring,station,layer,chamber,etaP),digi); 

      }
    }
  }
  
  // unPacker->readFedBuffers(*gemDigi, *gemPadDigi, *gemPadDigiCluster, *gemCoPadDigi
  // 			   *fed_buffers, theMapping, e);
  
  // put the digi data to the event  
  e.put(std::move(gemDigi), "MuonGEMDigis");
  // e.put(std::move(gemPadDigi), "MuonGEMPadDigis");
  // e.put(std::move(gemPadDigiCluster), "MuonGEMPadDigiClusters");
  // e.put(std::move(gemCoPadDigi), "CscTriggerPrimitiveDigis");
}


