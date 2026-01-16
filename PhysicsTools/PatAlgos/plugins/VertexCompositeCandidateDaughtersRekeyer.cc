#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace pat {
  class VertexCompositeCandidateDaughtersRekeyer : public edm::stream::EDProducer<> {
  public:
    explicit VertexCompositeCandidateDaughtersRekeyer(const edm::ParameterSet &iConfig);
    ~VertexCompositeCandidateDaughtersRekeyer() override;

    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    // configurables
    edm::EDGetTokenT<reco::VertexCompositePtrCandidateCollection> src_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcOriToken_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcNewToken_;
  };

}  // namespace pat

using namespace pat;

VertexCompositeCandidateDaughtersRekeyer::VertexCompositeCandidateDaughtersRekeyer(const edm::ParameterSet &iConfig)
    : src_(consumes<reco::VertexCompositePtrCandidateCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      pcOriToken_(
          consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesOri"))),
      pcNewToken_(
          consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))) {
  produces<reco::VertexCompositePtrCandidateCollection>();
}

VertexCompositeCandidateDaughtersRekeyer::~VertexCompositeCandidateDaughtersRekeyer() {}

void VertexCompositeCandidateDaughtersRekeyer::produce(edm::Event &iEvent, edm::EventSetup const &) {
  edm::Handle<reco::VertexCompositePtrCandidateCollection> src;
  iEvent.getByToken(src_, src);

  edm::Handle<pat::PackedCandidateCollection> pcOriHandle;
  iEvent.getByToken(pcOriToken_, pcOriHandle);

  edm::Handle<pat::PackedCandidateCollection> pcNewHandle;
  iEvent.getByToken(pcNewToken_, pcNewHandle);

  auto outPtrP = std::make_unique<reco::VertexCompositePtrCandidateCollection>();
  outPtrP->reserve(src->size());

  for (const auto &obj : *src) {
    // copy original object and append to vector
    outPtrP->emplace_back(obj);

    std::vector<reco::CandidatePtr> daughters = outPtrP->back().daughterPtrVector();
    outPtrP->back().clearDaughters();

    for (const reco::CandidatePtr &dau : daughters) {
      //
      // We check if this CandidatePtr points to a candidate in the original packedPFCandidates collection
      // This is needed because the CandidatePtr can point to a candidate in lostTracks collection
      //
      outPtrP->back().addDaughter((dau.id() == pcOriHandle.id()) ? edm::Ptr<reco::Candidate>(pcNewHandle, dau.key())
                                                                 : dau);
    }
  }
  iEvent.put(std::move(outPtrP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(VertexCompositeCandidateDaughtersRekeyer);
