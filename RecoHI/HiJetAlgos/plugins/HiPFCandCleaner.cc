#include "RecoHI/HiJetAlgos/plugins/HiPFCandCleaner.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

//
// constructors and destructor
//
HiPFCandCleaner::HiPFCandCleaner(const edm::ParameterSet& iConfig)
    : candidatesToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("candidatesSrc"))),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      absEtaMax_(iConfig.getParameter<double>("absEtaMax")) {
  produces<reco::PFCandidateCollection>("particleFlowCleaned");
}

HiPFCandCleaner::~HiPFCandCleaner() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void HiPFCandCleaner::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::PFCandidateCollection> candidates;
  iEvent.getByToken(candidatesToken_, candidates);

  auto prod = std::make_unique<reco::PFCandidateCollection>();

  for (auto const& cand : *candidates) {
    if (cand.pt() < ptMin_)
      continue;
    if (std::abs(cand.eta()) > absEtaMax_)
      continue;
    if (cand.particleId() != 1)
      continue;

    prod->push_back(cand);
  }

  iEvent.put(std::move(prod), "particleFlowCleaned");
}

void HiPFCandCleaner::beginJob() {}

void HiPFCandCleaner::endJob() {}

DEFINE_FWK_MODULE(HiPFCandCleaner);
