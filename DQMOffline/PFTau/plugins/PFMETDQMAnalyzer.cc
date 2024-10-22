#include "DQMOffline/PFTau/plugins/PFMETDQMAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DQMServices/Core/interface/DQMStore.h"

//
// -- Constructor
//
PFMETDQMAnalyzer::PFMETDQMAnalyzer(const edm::ParameterSet &parameterSet)

{
  pSet_ = parameterSet;
  inputLabel_ = pSet_.getParameter<edm::InputTag>("InputCollection");
  matchLabel_ = pSet_.getParameter<edm::InputTag>("MatchCollection");
  benchmarkLabel_ = pSet_.getParameter<std::string>("BenchmarkLabel");

  pfMETMonitor_.setParameters(parameterSet);

  myMET_ = consumes<edm::View<reco::MET>>(inputLabel_);
  myMatchedMET_ = consumes<edm::View<reco::MET>>(matchLabel_);

  std::string folder = benchmarkLabel_;

  subsystemname_ = "ParticleFlow";
  eventInfoFolder_ = subsystemname_ + "/" + folder;

  nBadEvents_ = 0;
}

//
// -- BookHistograms
//
void PFMETDQMAnalyzer::bookHistograms(DQMStore::IBooker &ibooker,
                                      edm::Run const & /* iRun */,
                                      edm::EventSetup const & /* iSetup */) {
  ibooker.setCurrentFolder(eventInfoFolder_);

  edm::LogInfo("PFMETDQMAnalyzer") << " PFMETDQMAnalyzer::bookHistograms "
                                   << "Histogram Folder path set to " << eventInfoFolder_;

  pfMETMonitor_.setup(ibooker, pSet_);
}

//
// -- Analyze
//
void PFMETDQMAnalyzer::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  edm::Handle<edm::View<reco::MET>> metCollection;
  iEvent.getByToken(myMET_, metCollection);

  edm::Handle<edm::View<reco::MET>> matchedMetCollection;
  iEvent.getByToken(myMatchedMET_, matchedMetCollection);

  if (metCollection.isValid() && matchedMetCollection.isValid()) {
    float maxRes = 0.0;
    float minRes = 99.99;
    pfMETMonitor_.fillOne((*metCollection)[0], (*matchedMetCollection)[0], minRes, maxRes);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFMETDQMAnalyzer);
