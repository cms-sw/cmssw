#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace pat {
  class PATPhotonCandidatesRekeyer : public edm::stream::EDProducer<> {
  public:
    explicit PATPhotonCandidatesRekeyer(const edm::ParameterSet &iConfig);
    ~PATPhotonCandidatesRekeyer() override;

    void produce(edm::Event &, const edm::EventSetup &) override;

  private:
    // configurables
    edm::EDGetTokenT<std::vector<pat::Photon>> src_;
    edm::EDGetTokenT<reco::CandidateView> pcNewCandViewToken_;
    edm::EDGetTokenT<pat::PackedCandidateCollection> pcNewToken_;
  };

}  // namespace pat

using namespace pat;

PATPhotonCandidatesRekeyer::PATPhotonCandidatesRekeyer(const edm::ParameterSet &iConfig):
  src_(consumes<std::vector<pat::Photon>>(iConfig.getParameter<edm::InputTag>("src"))),
  pcNewCandViewToken_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))),
  pcNewToken_(consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedPFCandidatesNew"))){
  produces<std::vector<pat::Photon>>();
}

PATPhotonCandidatesRekeyer::~PATPhotonCandidatesRekeyer() {}

void PATPhotonCandidatesRekeyer::produce(edm::Event &iEvent, edm::EventSetup const &) {
  edm::Handle<std::vector<pat::Photon>> src;
  iEvent.getByToken(src_, src);

  edm::Handle<reco::CandidateView> pcNewCandViewHandle;
  iEvent.getByToken(pcNewCandViewToken_, pcNewCandViewHandle);

  edm::Handle<pat::PackedCandidateCollection> pcNewHandle;
  iEvent.getByToken(pcNewToken_, pcNewHandle);

  auto outPtrP = std::make_unique<std::vector<pat::Photon>>();
  outPtrP->reserve(src->size());

  for (size_t i = 0; i < src->size(); ++i) {
    // copy original pat object and append to vector
    outPtrP->emplace_back((*src)[i]);

    std::vector<unsigned int> keys;
    for (const edm::Ref<pat::PackedCandidateCollection> &ref : outPtrP->back().associatedPackedPFCandidates()){
      keys.push_back(ref.key());
    };
    outPtrP->back().setAssociatedPackedPFCandidates(
      edm::RefProd<pat::PackedCandidateCollection>(pcNewHandle),
      keys.begin(), keys.end()
    );
    if (keys.size() == 1) {
      outPtrP->back().refToOrig_ = outPtrP->back().sourceCandidatePtr(0);
    } else {
      outPtrP->back().refToOrig_ = reco::CandidatePtr(pcNewHandle.id());
    }
  }
  iEvent.put(std::move(outPtrP));
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPhotonCandidatesRekeyer);
