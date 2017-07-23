/** \file
 *  \author J. Lee - UoS
 */

#include "EventFilter/GEMRawToDigi/src/GEMDigiToRawModule.h"
#include "EventFilter/GEMRawToDigi/src/GEMDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CondFormats/DataRecord/interface/GEMChamberMapRcd.h"

GEMDigiToRawModule::GEMDigiToRawModule(const edm::ParameterSet & pset): 
  packer(new GEMDigiToRaw(pset))
{
  event_type_ = pset.getParameter<int>("eventType");
  digi_token = consumes<GEMDigiCollection>( pset.getParameter<edm::InputTag>("gemDigi") );
  padDigi_token = consumes<GEMPadDigiCollection>( pset.getParameter<edm::InputTag>("gemPadDigi") );
  padDigiCluster_token = consumes<GEMPadDigiClusterCollection>( pset.getParameter<edm::InputTag>("gemPadDigiCluster") );
  coPadDigi_token = consumes<GEMCoPadDigiCollection>( pset.getParameter<edm::InputTag>("gemCoPadDigi") );

  produces<FEDRawDataCollection>("GEMRawData"); 

}


GEMDigiToRawModule::~GEMDigiToRawModule(){
  delete packer;
}

void GEMDigiToRawModule::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gemDigi", edm::InputTag("simMuonGEMDigis"));
  desc.add<edm::InputTag>("gemPadDigi", edm::InputTag("simMuonGEMPadDigis"));
  desc.add<edm::InputTag>("gemPadDigiCluster", edm::InputTag("simMuonGEMPadDigiClusters"));
  desc.add<edm::InputTag>("gemCoPadDigi", edm::InputTag("simCscTriggerPrimitiveDigis"));
 
  descriptions.add("gemPacker", desc);
}


void GEMDigiToRawModule::produce( edm::Event & e, const edm::EventSetup& c ){
  ///reverse mapping for packer
  edm::ESHandle<GEMChamberMap> gemChamberMap;
  c.get<GEMChamberMapRcd>().get(gemChamberMap); 
  const GEMChamberMap* theMapping = gemChamberMap.product();

  auto fed_buffers = std::make_unique<FEDRawDataCollection>();

  // Take digis from the event
  edm::Handle<GEMDigiCollection> gemDigi;
  edm::Handle<GEMPadDigiCollection> gemPadDigi;
  edm::Handle<GEMPadDigiClusterCollection> gemPadDigiCluster;
  edm::Handle<GEMCoPadDigiCollection> gemCoPadDigi;

  e.getByToken( digi_token, gemDigi );
  e.getByToken( padDigi_token, gemPadDigi );
  e.getByToken( padDigiCluster_token, gemPadDigiCluster );
  e.getByToken( coPadDigi_token, gemCoPadDigi );


  for (unsigned int id=FEDNumbering::MINGEMFEDID; id<=FEDNumbering::MAXGEMFEDID; ++id){ 

    FEDHeader::set(data.data() + size * 8, event_type_, e.id().event(), e.bunchCrossing(), fed_amcs.first);

  std::vector<unsigned char> byteVec;  
  // make FEROL headers
  uint64_t m_word;

  // 
  m_AMC13Event = new AMC13Event();
  m_AMC13Event->setCDFHeader(m_word);
  m_AMC13Event->setAMC13header(m_word);

  for (unsigned short i = 0; i < m_AMC13Event->nAMC(); i++){
    std::fread(&m_word, sizeof(uint64_t), 1, m_file);
    if(verbose_)  printf("%016lX\n", m_word);
    GEMUnpacker::ByteVector(byteVec, m_word);
    m_AMC13Event->addAMCheader(m_word);
  }

  // Create the packed data
  packer->createFedBuffers(*gemDigi, *gemPadDigi, *gemPadDigiCluster, *gemCoPadDigi
                           *(fed_buffers.get()), theMapping, e);
  
  // put the raw data to the event
  e.put(std::move(fed_buffers), "GEMRawData");
}


