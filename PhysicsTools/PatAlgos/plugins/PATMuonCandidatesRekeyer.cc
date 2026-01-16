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

    //
    std::vector<unsigned int> keys;
    for (size_t ic = 0; ic < outPtrP->back().numberOfSourceCandidatePtrs(); ++ic) {
      const reco::CandidatePtr &candPtr = outPtrP->back().sourceCandidatePtr(ic);
      if (candPtr.isNonnull()) {
        keys.push_back(candPtr.key());
      }
    }
    if (keys.size() == 1) {
      outPtrP->back().refToOrig_ = reco::CandidatePtr(pcNewCandViewHandle, keys[0]);
    }
  }

  iEvent.put(std::move(outPtrP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATMuonCandidatesRekeyer);
