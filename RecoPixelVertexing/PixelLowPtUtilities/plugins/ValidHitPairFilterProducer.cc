#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/ValidHitPairFilter.h"

class ValidHitPairFilterProducer: public edm::global::EDProducer<> {
public:
  explicit ValidHitPairFilterProducer(const edm::ParameterSet& iConfig);
  ~ValidHitPairFilterProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
};

ValidHitPairFilterProducer::ValidHitPairFilterProducer(const edm::ParameterSet& iConfig) {
  produces<PixelTrackFilter>();
}

void ValidHitPairFilterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("validHitPairFilter", desc);
}

ValidHitPairFilterProducer::~ValidHitPairFilterProducer() {}

void ValidHitPairFilterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto impl = std::make_unique<ValidHitPairFilter>(iSetup);
  auto prod = std::make_unique<PixelTrackFilter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(ValidHitPairFilterProducer);
