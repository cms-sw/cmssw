#include <memory>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

class HiSignalParticleProducer : public edm::global::EDProducer<> {
public:
  explicit HiSignalParticleProducer(const edm::ParameterSet&);
  ~HiSignalParticleProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<edm::View<reco::GenParticle> > genParticleSrc_;
};

HiSignalParticleProducer::HiSignalParticleProducer(const edm::ParameterSet& iConfig)
    : genParticleSrc_(consumes<edm::View<reco::GenParticle> >(iConfig.getParameter<edm::InputTag>("src"))) {
  std::string alias = (iConfig.getParameter<edm::InputTag>("src")).label();
  produces<reco::GenParticleCollection>().setBranchAlias(alias);
}

void HiSignalParticleProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto signalGenParticles = std::make_unique<reco::GenParticleCollection>();

  edm::Handle<edm::View<reco::GenParticle> > genParticles;
  iEvent.getByToken(genParticleSrc_, genParticles);

  for (const reco::GenParticle& genParticle : *genParticles) {
    if (genParticle.collisionId() == 0) {
      signalGenParticles->push_back(genParticle);
    }
  }

  iEvent.put(std::move(signalGenParticles));
}

void HiSignalParticleProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Selects genParticles from collision id = 0");
  desc.add<edm::InputTag>("src", edm::InputTag("genParticles"));
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HiSignalParticleProducer);
