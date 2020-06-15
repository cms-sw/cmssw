#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFiller.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class HiPFCandCleaner : public edm::global::EDProducer<> {
public:
  explicit HiPFCandCleaner(const edm::ParameterSet&);

  // class methods

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::PFCandidateCollection> candidatesToken_;

  double ptMin_;
  double absEtaMax_;
};
//
// constructors and destructor
//
HiPFCandCleaner::HiPFCandCleaner(const edm::ParameterSet& iConfig)
    : candidatesToken_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("candidatesSrc"))),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      absEtaMax_(iConfig.getParameter<double>("absEtaMax")) {
  produces<reco::PFCandidateCollection>("particleFlowCleaned");
}

// ------------ method called to for each event  ------------
void HiPFCandCleaner::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
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

DEFINE_FWK_MODULE(HiPFCandCleaner);
