#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class EventSetup;
}  // namespace edm

class PFConcretePFCandidateProducer : public edm::stream::EDProducer<> {
public:
  explicit PFConcretePFCandidateProducer(const edm::ParameterSet&);
  ~PFConcretePFCandidateProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::PFCandidateCollection> inputColl_;
};

DEFINE_FWK_MODULE(PFConcretePFCandidateProducer);

void PFConcretePFCandidateProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", {"particleFlow"});
  descriptions.addWithDefaultLabel(desc);
}

PFConcretePFCandidateProducer::PFConcretePFCandidateProducer(const edm::ParameterSet& iConfig)
    : inputColl_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("src"))) {
  // register products
  produces<reco::PFCandidateCollection>();
}

PFConcretePFCandidateProducer::~PFConcretePFCandidateProducer() {}

void PFConcretePFCandidateProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& inputColl = iEvent.getHandle(inputColl_);

  if (!inputColl.isValid()) {
    // nothing ... I guess we prefer to send an exception in the next lines
  }

  auto outputColl = std::make_unique<reco::PFCandidateCollection>();
  outputColl->resize(inputColl->size());

  for (unsigned int iCopy = 0; iCopy != inputColl->size(); ++iCopy) {
    const reco::PFCandidate& pf = (*inputColl)[iCopy];
    (*outputColl)[iCopy] = pf;
    //dereferenced internally the ref and hardcopy the value
    (*outputColl)[iCopy].setVertex(pf.vertex());
    //math::XYZPoint(pf.vx(),pf.vy(),pf.vz()));
  }

  iEvent.put(std::move(outputColl));
}
