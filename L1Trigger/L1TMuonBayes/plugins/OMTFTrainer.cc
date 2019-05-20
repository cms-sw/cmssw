#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFConfiguration.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinput.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/OMTFinputMaker.h>
#include <L1Trigger/L1TMuonBayes/interface/Omtf/XMLConfigWriter.h>
#include <L1Trigger/L1TMuonBayes/interface/OmtfPatternGeneration/OMTFConfigMaker.h>
#include <iostream>
#include <strstream>
#include <iomanip>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"

#include "L1Trigger/L1TMuonBayes/plugins/OMTFTrainer.h"
#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include "SimDataFormats/Track/interface/SimTrack.h"

#include "Math/VectorUtil.h"

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include <TH2F.h>
#include "TFile.h"


OMTFTrainer::OMTFTrainer(const edm::ParameterSet& cfg):
theConfig(cfg),
g4SimTrackSrc(cfg.getParameter<edm::InputTag>("g4SimTrackSrc")), m_Reconstruction(cfg, muStubsInputTokens) {

  produces<l1t::RegionalMuonCandBxCollection >("OMTF");

  muStubsInputTokens.inputTokenDTPh = consumes<L1MuDTChambPhContainer>(theConfig.getParameter<edm::InputTag>("srcDTPh"));
  muStubsInputTokens.inputTokenDTTh = consumes<L1MuDTChambThContainer>(theConfig.getParameter<edm::InputTag>("srcDTTh"));
  muStubsInputTokens.inputTokenCSC = consumes<CSCCorrelatedLCTDigiCollection>(theConfig.getParameter<edm::InputTag>("srcCSC"));
  muStubsInputTokens.inputTokenRPC = consumes<RPCDigiCollection>(theConfig.getParameter<edm::InputTag>("srcRPC"));

  inputTokenSimHit = consumes<edm::SimTrackContainer>(theConfig.getParameter<edm::InputTag>("g4SimTrackSrc"));

  ptDist = new TH1I("ptDist", "ptDist", 200, -0.5, 200-0.5);

  etaCutFrom = theConfig.getParameter<double>("etaCutFrom");
  etaCutTo = theConfig.getParameter<double>("etaCutTo");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
OMTFTrainer::~OMTFTrainer(){

}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFTrainer::beginRun(edm::Run const& run, edm::EventSetup const& iSetup) {
  m_Reconstruction.beginRun(run, iSetup);
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFTrainer::beginJob(){
  m_Reconstruction.beginJob();
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////  
void OMTFTrainer::endJob(){
  m_Reconstruction.endJob();
}

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
void OMTFTrainer::produce(edm::Event& iEvent, const edm::EventSetup& evSetup){
  const SimTrack * simMuon = 0;
  edm::Handle<edm::SimTrackContainer> simTks;
  iEvent.getByToken(inputTokenSimHit,simTks);

  int muCnt = 0;
  for (std::vector<SimTrack>::const_iterator it=simTks->begin(); it< simTks->end(); it++) {
    const SimTrack & aTrack = *it;
    if ( !(aTrack.type() == 13 || aTrack.type() == -13) )
      continue;
    muCnt++;
    if ( !simMuon || aTrack.momentum().pt() > simMuon->momentum().pt())
      simMuon = &aTrack;
  }

  /*cout<<__FUNCTION__<<":"<<__LINE__<<" mmuCnt "<<muCnt;
  if(muCnt > 0) {
    cout<<" simMuon pt "<<simMuon->momentum().pt()<<" eta "<<simMuon->momentum().eta()<<std::endl;
  }*/

  if(muCnt != 1 || ( abs(simMuon->momentum().eta() ) < etaCutFrom || abs(simMuon->momentum().eta() ) > etaCutTo ) ) {
    std::unique_ptr<l1t::RegionalMuonCandBxCollection> candidates(new l1t::RegionalMuonCandBxCollection);
    iEvent.put(std::move(candidates), "OMTF");
    return;
  }

  std::ostringstream str;
  std::unique_ptr<l1t::RegionalMuonCandBxCollection > candidates = m_Reconstruction.reconstruct(iEvent, evSetup);

  int bx = 0;
  edm::LogInfo("OMTFOMTFTrainer")<<" Number of candidates: "<<candidates->size(bx);

  iEvent.put(std::move(candidates), "OMTF");
}
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////  
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(OMTFTrainer);
