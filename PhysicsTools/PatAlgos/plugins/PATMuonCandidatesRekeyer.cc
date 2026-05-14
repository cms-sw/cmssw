#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include <algorithm>
#include <cmath>

namespace {
  bool sameFloat(float lhs, float rhs, float relTol, float absTol = 1.e-7f) {
    return std::abs(lhs - rhs) <= std::max(absTol, relTol * std::max(std::abs(lhs), std::abs(rhs)));
  }

  bool isConsistentPackedCandidate(const reco::CandidatePtr &candPtr,
                                   const pat::PackedCandidateCollection &packedCands,
                                   unsigned int key) {
    if (candPtr.isNull() || !candPtr.isAvailable() || key >= packedCands.size()) {
      return false;
    }

    const auto *oldPacked = dynamic_cast<const pat::PackedCandidate *>(candPtr.get());
    if (oldPacked == nullptr) {
      return false;
    }

    const auto &newPacked = packedCands[key];
    return oldPacked->charge() == newPacked.charge() && oldPacked->pdgId() == newPacked.pdgId() &&
           sameFloat(oldPacked->pt(), newPacked.pt(), 1.e-5f) && sameFloat(oldPacked->eta(), newPacked.eta(), 1.e-6f) &&
           sameFloat(oldPacked->phi(), newPacked.phi(), 1.e-6f) &&
           sameFloat(oldPacked->mass(), newPacked.mass(), 1.e-5f);
  }
}  // namespace

namespace pat {
  class PATMuonCandidatesRekeyer : public edm::stream::EDProducer<> {
  public:
    explicit PATMuonCandidatesRekeyer(const edm::ParameterSet &iConfig);
    ~PATMuonCandidatesRekeyer() override;

    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    // configurables
    edm::EDGetTokenT<std::vector<pat::Muon>> src_;
    edm::EDGetTokenT<reco::CandidateView> pcNewCandViewToken_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcNewToken_;
  };

}  // namespace pat

using namespace pat;

PATMuonCandidatesRekeyer::PATMuonCandidatesRekeyer(const edm::ParameterSet &iConfig)
    : src_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("src"))),
      pcNewCandViewToken_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))),
      pcNewToken_(
          consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))) {
  produces<std::vector<pat::Muon>>();
}

PATMuonCandidatesRekeyer::~PATMuonCandidatesRekeyer() {}

void PATMuonCandidatesRekeyer::produce(edm::Event &iEvent, edm::EventSetup const &) {
  edm::Handle<std::vector<pat::Muon>> src;
  iEvent.getByToken(src_, src);

  edm::Handle<reco::CandidateView> pcNewCandViewHandle;
  iEvent.getByToken(pcNewCandViewToken_, pcNewCandViewHandle);

  edm::Handle<pat::PackedCandidateCollection> pcNewHandle;
  iEvent.getByToken(pcNewToken_, pcNewHandle);

  auto outPtrP = std::make_unique<std::vector<pat::Muon>>();
  outPtrP->reserve(src->size());

  for (const auto &obj : *src) {
    // copy original pat object and append to vector
    outPtrP->emplace_back(obj);

    auto &muon = outPtrP->back();
    const reco::CandidatePtr *sourcePtr = nullptr;

    if (muon.refToOrig_.isNonnull()) {
      sourcePtr = &muon.refToOrig_;
    } else if (muon.numberOfSourceCandidatePtrs() == 1) {
      const reco::CandidatePtr &candPtr = muon.sourceCandidatePtr(0);
      sourcePtr = candPtr.isNonnull() ? &candPtr : nullptr;
    }

    if (sourcePtr != nullptr) {
      const unsigned int key = sourcePtr->key();
      if (isConsistentPackedCandidate(*sourcePtr, *pcNewHandle, key)) {
        muon.refToOrig_ = reco::CandidatePtr(pcNewCandViewHandle, key);
      }
    }
  }

  iEvent.put(std::move(outPtrP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATMuonCandidatesRekeyer);
