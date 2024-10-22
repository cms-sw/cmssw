#include <memory>
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace pat {

  class PackedGenParticleSignalProducer : public edm::global::EDProducer<> {
  public:
    explicit PackedGenParticleSignalProducer(const edm::ParameterSet& iConfig)
        : genParticleToken_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genParticles"))),
          assoToken_(consumes<edm::Association<std::vector<pat::PackedGenParticle>>>(
              iConfig.getParameter<edm::InputTag>("packedGenParticles"))) {
      produces<pat::PackedGenParticleRefVector>();
    }
    ~PackedGenParticleSignalProducer() override = default;

    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    const edm::EDGetTokenT<reco::GenParticleCollection> genParticleToken_;
    const edm::EDGetTokenT<edm::Association<std::vector<pat::PackedGenParticle>>> assoToken_;
  };

}  // namespace pat

void pat::PackedGenParticleSignalProducer::produce(edm::StreamID iID,
                                                   edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup) const {
  const auto& genParticles = iEvent.getHandle(genParticleToken_);
  const auto& orig2packed = iEvent.get(assoToken_);

  auto signalGenParticleRefs = std::make_unique<pat::PackedGenParticleRefVector>();

  for (auto it = genParticles->begin(); it != genParticles->end(); ++it) {
    const auto& orig = reco::GenParticleRef(genParticles, it - genParticles->begin());
    if (orig->collisionId() != 0)
      continue;
    const auto& packed = orig2packed[orig];
    if (packed.isNonnull()) {
      signalGenParticleRefs->push_back(packed);
    }
  }

  iEvent.put(std::move(signalGenParticleRefs));
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
