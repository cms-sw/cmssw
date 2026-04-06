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
    bool haveKey = false;
    unsigned int key = 0;

    // Muons can expose both pfCandidateRef_ and refToOrig_, so requiring a
    // unique sourceCandidatePtr() is too strict for the packed-candidate rekey.
    if (muon.refToOrig_.isNonnull()) {
      key = muon.refToOrig_.key();
      haveKey = true;
    } else if (muon.numberOfSourceCandidatePtrs() == 1) {
      const reco::CandidatePtr &candPtr = muon.sourceCandidatePtr(0);
      if (candPtr.isNonnull()) {
        key = candPtr.key();
        haveKey = true;
      }
    }

    if (haveKey && key < pcNewHandle->size()) {
      muon.refToOrig_ = reco::CandidatePtr(pcNewCandViewHandle, key);
    }
  }

  iEvent.put(std::move(outPtrP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATMuonCandidatesRekeyer);
