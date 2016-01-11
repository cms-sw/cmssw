#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

class TrackingParticleConversionSelector: public edm::stream::EDProducer<> {
public:
  TrackingParticleConversionSelector(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

private:
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
};


TrackingParticleConversionSelector::TrackingParticleConversionSelector(const edm::ParameterSet& iConfig):
  tpToken_(consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("src")))
{
  produces<TrackingParticleCollection>();
}

void TrackingParticleConversionSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("mix", "MergedTrackTruth"));
  descriptions.add("trackingParticleConversionSelector", desc);
}

void TrackingParticleConversionSelector::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TrackingParticleCollection> h_tps;
  iEvent.getByToken(tpToken_, h_tps);

  // Copy TPs because currently we can't process RefVectors of TPs in downstream
  auto ret = std::make_unique<TrackingParticleCollection>();

  // Logic is similar to Validation/RecoEgamma/plugins/PhotonValidator.cc
  // and RecoEgamma/EgammaMCTools/src/PhotonMCTruthFinder.cc,
  // but implemented purely in terms of TrackingParticles (simpler and works for pileup too)
  for(const auto& tp: *h_tps) {
    if(tp.pdgId() == 22) {
      for(const auto& vertRef: tp.decayVertices()) {
        for(const auto& tpRef: vertRef->daughterTracks()) {
          if(std::abs(tpRef->pdgId()) == 11) {
            ret->push_back(*tpRef);
          }
        }
      }
    }
  }

  iEvent.put(std::move(ret));
}

DEFINE_FWK_MODULE(TrackingParticleConversionSelector);
