
#include <memory>

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace pat {

  class PackedGenParticleSignalProducer : public edm::stream::EDProducer<> {
  public:
    explicit PackedGenParticleSignalProducer(const edm::ParameterSet& iConfig)
        : packedGenParticleToken_(
              consumes<pat::PackedGenParticleCollection>(iConfig.getParameter<edm::InputTag>("packedGenParticles"))),
          genParticleToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
          assoToken_(consumes<edm::Association<std::vector<pat::PackedGenParticle>>>(
              iConfig.getParameter<edm::InputTag>("packedGenParticles"))) {
      produces<pat::PackedGenParticleRefVector>();
    }
    ~PackedGenParticleSignalProducer() override = default;

    void produce(edm::Event&, const edm::EventSetup&) override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<pat::PackedGenParticleCollection> packedGenParticleToken_;
    const edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
    const edm::EDGetTokenT<edm::Association<std::vector<pat::PackedGenParticle>>> assoToken_;
  };

}  // namespace pat

void pat::PackedGenParticleSignalProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& packedGenParticles = iEvent.getHandle(packedGenParticleToken_);
  const auto& genParticles = iEvent.getHandle(genParticleToken_);
  const auto& packed2Orig = iEvent.get(assoToken_);

  pat::PackedGenParticleRefVector* signalGenParticleRefs = new pat::PackedGenParticleRefVector();

  for (size_t i = 0; i < genParticles->size(); ++i) {
    const auto& orig = reco::GenParticleRef(genParticles, i);

    if (orig.isNonnull()) {
      if (orig->collisionId() == 0) {
        const auto& packed = packed2Orig[orig];
        if (packed.isNonnull())
          signalGenParticleRefs->push_back(packed);
      }
    }
  }

  std::unique_ptr<pat::PackedGenParticleRefVector> ptr(signalGenParticleRefs);
  iEvent.put(std::move(ptr));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void pat::PackedGenParticleSignalProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("genParticles", edm::InputTag("genParticles"))->setComment("genParticles input collection");
  desc.add<edm::InputTag>("packedGenParticles", edm::InputTag("packedGenParticles"))
      ->setComment("packedGenParticles input collection");
  descriptions.add("packedGenParticlesSignal", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(PackedGenParticleSignalProducer);
