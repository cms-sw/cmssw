#include "DQMOffline/PFTau/plugins/PFCandidateManagerAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace reco;
using namespace edm;
using namespace std;

PFCandidateManagerAnalyzer::PFCandidateManagerAnalyzer(const edm::ParameterSet &parameterSet)
    : BenchmarkAnalyzer(parameterSet),
      PFCandidateManager(parameterSet.getParameter<double>("dRMax"),
                         parameterSet.getParameter<bool>("matchCharge"),
                         (Benchmark::Mode)parameterSet.getParameter<int>("mode")),
      matchLabel_(parameterSet.getParameter<InputTag>("MatchCollection")) {
  setRange(parameterSet.getParameter<double>("ptMin"),
           parameterSet.getParameter<double>("ptMax"),
           parameterSet.getParameter<double>("etaMin"),
           parameterSet.getParameter<double>("etaMax"),
           parameterSet.getParameter<double>("phiMin"),
           parameterSet.getParameter<double>("phiMax"));

  myColl_ = consumes<PFCandidateCollection>(inputLabel_);
  myMatchColl_ = consumes<View<Candidate>>(matchLabel_);
}

void PFCandidateManagerAnalyzer::bookHistograms(DQMStore::IBooker &ibooker,
                                                edm::Run const &iRun,
                                                edm::EventSetup const &iSetup) {
  BenchmarkAnalyzer::bookHistograms(ibooker, iRun, iSetup);
  setup(ibooker);
}

void PFCandidateManagerAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  Handle<PFCandidateCollection> collection;
  iEvent.getByToken(myColl_, collection);

  Handle<View<Candidate>> matchCollection;
  iEvent.getByToken(myMatchColl_, matchCollection);

  fill(*collection, *matchCollection);
}
