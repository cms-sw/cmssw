#include "DQMOffline/PFTau/plugins/PFCandidateDQMAnalyzer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DQMServices/Core/interface/DQMStore.h"

//
// -- Constructor
//
PFCandidateDQMAnalyzer::PFCandidateDQMAnalyzer(const edm::ParameterSet &parameterSet)

{
  pSet_ = parameterSet;
  inputLabel_ = pSet_.getParameter<edm::InputTag>("InputCollection");
  matchLabel_ = pSet_.getParameter<edm::InputTag>("MatchCollection");
  benchmarkLabel_ = pSet_.getParameter<std::string>("BenchmarkLabel");
  createEfficiencyHistos_ = pSet_.getParameter<bool>("CreateEfficiencyHistos");

  pfCandidateMonitor_.setParameters(parameterSet);

  myCand_ = consumes<edm::View<reco::Candidate>>(inputLabel_);
  myMatchedCand_ = consumes<edm::View<reco::Candidate>>(matchLabel_);

  std::string folder = benchmarkLabel_;

  subsystemname_ = "ParticleFlow";
  eventInfoFolder_ = subsystemname_ + "/" + folder;

  nBadEvents_ = 0;
}

//
// -- BookHistograms
//
void PFCandidateDQMAnalyzer::bookHistograms(DQMStore::IBooker &ibooker,
                                            edm::Run const & /* iRun */,
                                            edm::EventSetup const & /* iSetup */) {
  ibooker.setCurrentFolder(eventInfoFolder_);

  edm::LogInfo("PFCandidateDQMAnalyzer") << " PFCandidateDQMAnalyzer::bookHistograms "
                                         << "Histogram Folder path set to " << eventInfoFolder_;

  pfCandidateMonitor_.setup(ibooker, pSet_);
}

//
// -- Analyze
//
void PFCandidateDQMAnalyzer::analyze(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  edm::Handle<edm::View<reco::Candidate>> candCollection;
  edm::Handle<edm::View<reco::Candidate>> matchedCandCollection;
  if (!createEfficiencyHistos_) {
    iEvent.getByToken(myCand_, candCollection);
    iEvent.getByToken(myMatchedCand_, matchedCandCollection);
  } else {
    iEvent.getByToken(myMatchedCand_, candCollection);
    iEvent.getByToken(myCand_, matchedCandCollection);
  }

  float maxRes = 0.0;
  float minRes = 99.99;
  if (candCollection.isValid() && matchedCandCollection.isValid()) {
    pfCandidateMonitor_.fill(*candCollection, *matchedCandCollection, minRes, maxRes, pSet_);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateDQMAnalyzer);
