#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace pat {
  class PATPackedCandidateUpdater : public edm::stream::EDProducer<> {
  public:
    explicit PATPackedCandidateUpdater(const edm::ParameterSet&);
    ~PATPackedCandidateUpdater() override {}

    void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    edm::EDGetTokenT<std::vector<pat::PackedCandidate>> candsToken_;

    bool updatePuppiWeights_;
    edm::EDGetTokenT<edm::ValueMap<float>> puppiWeightToken_;
    edm::EDGetTokenT<edm::ValueMap<float>> puppiWeightNoLepToken_;
  };
}  // namespace pat

using namespace pat;

PATPackedCandidateUpdater::PATPackedCandidateUpdater(const edm::ParameterSet& iConfig)
    : candsToken_(consumes<std::vector<pat::PackedCandidate>>(iConfig.getParameter<edm::InputTag>("src"))),
      updatePuppiWeights_(iConfig.getParameter<bool>("updatePuppiWeights")),
      puppiWeightToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiWeight"))),
      puppiWeightNoLepToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("puppiWeightNoLep"))) {
  produces<std::vector<pat::PackedCandidate>>();
}

void PATPackedCandidateUpdater::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<std::vector<pat::PackedCandidate>> cands;
  iEvent.getByToken(candsToken_, cands);

  edm::Handle<edm::ValueMap<float>> puppiWeight;
  edm::Handle<edm::ValueMap<float>> puppiWeightNoLep;
  if (updatePuppiWeights_) {
    iEvent.getByToken(puppiWeightToken_, puppiWeight);
    iEvent.getByToken(puppiWeightNoLepToken_, puppiWeightNoLep);
  }

  auto outPtrP = std::make_unique<std::vector<pat::PackedCandidate>>();
  outPtrP->reserve(cands->size());

  for (size_t ic = 0; ic < cands->size(); ++ic) {
    // copy original pat::PackedCandidate and append to vector
    outPtrP->emplace_back((*cands)[ic]);

    // Retrieve puppi weights from edm::ValueMap
    pat::PackedCandidateRef pkref(cands, ic);

    if (updatePuppiWeights_) {
      float puppiWeightVal = (*puppiWeight)[pkref];
      float puppiWeightNoLepVal = (*puppiWeightNoLep)[pkref];
      outPtrP->back().setPuppiWeight(puppiWeightVal, puppiWeightNoLepVal);
    }
  }

  iEvent.put(std::move(outPtrP));
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedCandidateUpdater);
