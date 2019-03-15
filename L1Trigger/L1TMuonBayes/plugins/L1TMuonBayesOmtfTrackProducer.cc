#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinput.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFProcessor.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/XMLConfigWriter.h>
#include <L1Trigger/L1TMuonBayes/plugins/L1TMuonBayesOmtfTrackProducer.h>
#include <iostream>
#include <strstream>
#include <vector>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

L1TMuonBayesOmtfTrackProducer::L1TMuonBayesOmtfTrackProducer(const edm::ParameterSet& cfg)
  :theConfig(cfg), m_Reconstruction(cfg) {

  produces<l1t::RegionalMuonCandBxCollection >("OMTF");

  inputTokenDTPh = consumes<L1MuDTChambPhContainer>(theConfig.getParameter<edm::InputTag>("srcDTPh"));
  inputTokenDTTh = consumes<L1MuDTChambThContainer>(theConfig.getParameter<edm::InputTag>("srcDTTh"));
  inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(theConfig.getParameter<edm::InputTag>("srcCSC"));
  inputTokenRPC = consumes<RPCDigiCollection>(theConfig.getParameter<edm::InputTag>("srcRPC"));

  inputTokenSimHit = consumes<edm::SimTrackContainer>(theConfig.getParameter<edm::InputTag>("g4SimTrackSrc")); //TODO remove
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
L1TMuonBayesOmtfTrackProducer::~L1TMuonBayesOmtfTrackProducer(){  
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesOmtfTrackProducer::beginJob(){

  m_Reconstruction.beginJob();

}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesOmtfTrackProducer::endJob(){

  m_Reconstruction.endJob();

}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesOmtfTrackProducer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup){

  m_Reconstruction.beginRun(run, iSetup);
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void L1TMuonBayesOmtfTrackProducer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup){

  std::ostringstream str;
  
  std::unique_ptr<l1t::RegionalMuonCandBxCollection > candidates = m_Reconstruction.reconstruct(iEvent, evSetup);

  iEvent.put(std::move(candidates), "OMTF");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonBayesOmtfTrackProducer);
