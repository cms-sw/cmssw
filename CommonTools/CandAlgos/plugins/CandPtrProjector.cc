#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

class CandPtrProjector : public edm::global::EDProducer<> {
public:
  explicit CandPtrProjector(edm::ParameterSet const& iConfig);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::View<reco::Candidate>> candSrcToken_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> vetoSrcToken_;
  bool useDeltaRforFootprint_;
  bool extendVetoBySingleSourcePtr_;
};

CandPtrProjector::CandPtrProjector(edm::ParameterSet const& iConfig)
    : candSrcToken_{consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("src"))},
      vetoSrcToken_{consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("veto"))},
      useDeltaRforFootprint_(iConfig.getParameter<bool>("useDeltaRforFootprint")),
      extendVetoBySingleSourcePtr_(iConfig.getParameter<bool>("extendVetoBySingleSourcePtr")) {
  produces<edm::PtrVector<reco::Candidate>>();
}

void CandPtrProjector::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  using namespace edm;
  Handle<View<reco::Candidate>> vetoes;
  iEvent.getByToken(vetoSrcToken_, vetoes);

  auto result = std::make_unique<PtrVector<reco::Candidate>>();
  std::set<reco::CandidatePtr> vetoedPtrs;
  for (auto const& veto : *vetoes) {
    auto const n = veto.numberOfSourceCandidatePtrs();
    for (size_t j{}; j < n; ++j) {
      vetoedPtrs.insert(veto.sourceCandidatePtr(j));
    }
  }

  Handle<View<reco::Candidate>> cands;
  iEvent.getByToken(candSrcToken_, cands);
  for (size_t i{}; i < cands->size(); ++i) {
    auto const c = cands->ptrAt(i);
    if (vetoedPtrs.find(c) == vetoedPtrs.cend()) {
      bool addcand = true;
      if ((extendVetoBySingleSourcePtr_) && (c->numberOfSourceCandidatePtrs() == 1))
        addcand = (vetoedPtrs.find(c->sourceCandidatePtr(0)) == vetoedPtrs.cend());
      if (useDeltaRforFootprint_)
        for (const auto& it : vetoedPtrs)
          if (it.isNonnull() && it.isAvailable() && reco::deltaR2(*it, *c) < 0.00000025) {
            addcand = false;
            break;
          }
      if (addcand)
        result->push_back(c);
    }
  }
  iEvent.put(std::move(result));
}

void CandPtrProjector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src");
  desc.add<edm::InputTag>("veto");
  desc.add<bool>("useDeltaRforFootprint", false);
  desc.add<bool>("extendVetoBySingleSourcePtr", true);
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(CandPtrProjector);
