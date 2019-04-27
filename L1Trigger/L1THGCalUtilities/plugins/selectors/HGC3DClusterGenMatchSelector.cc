#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"

namespace l1t {
  class HGC3DClusterGenMatchSelector : public edm::stream::EDProducer<> {
  public:
    explicit HGC3DClusterGenMatchSelector(const edm::ParameterSet &);
    ~HGC3DClusterGenMatchSelector() override {}

  private:
    edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> src_;
    edm::EDGetToken genParticleSrc_;
    double dR_;
    void produce(edm::Event &, const edm::EventSetup &) override;

  };  // class
}  // namespace l1t

l1t::HGC3DClusterGenMatchSelector::HGC3DClusterGenMatchSelector(const edm::ParameterSet &iConfig)
    : src_(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      genParticleSrc_(consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("genSrc"))),
      dR_(iConfig.getParameter<double>("dR")) {
  produces<l1t::HGCalMulticlusterBxCollection>();
}

void l1t::HGC3DClusterGenMatchSelector::produce(edm::Event &iEvent, const edm::EventSetup &) {
  auto out = std::make_unique<l1t::HGCalMulticlusterBxCollection>();

  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters;
  iEvent.getByToken(src_, multiclusters);

  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genParticleSrc_, genParticles);

  for (int bx = multiclusters->getFirstBX(); bx <= multiclusters->getLastBX(); ++bx) {
    for (auto it = multiclusters->begin(bx), ed = multiclusters->end(bx); it != ed; ++it) {
      const auto &multicluster = *it;
      for (const auto &particle : *genParticles) {
        if (particle.status() != 1)
          continue;
        if (deltaR(multicluster, particle) < dR_) {
          out->push_back(bx, multicluster);
          break;  // don't save duplicate multiclusters!
        }
      }
    }
  }

  iEvent.put(std::move(out));
}
using l1t::HGC3DClusterGenMatchSelector;
DEFINE_FWK_MODULE(HGC3DClusterGenMatchSelector);
