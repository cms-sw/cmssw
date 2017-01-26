#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackFitter.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class TrackFitterProducer: public edm::global::EDProducer<> {
public:
  explicit TrackFitterProducer(const edm::ParameterSet& iConfig);
  ~TrackFitterProducer() {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  std::string theTTRHBuilderName;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;
};

TrackFitterProducer::TrackFitterProducer(const edm::ParameterSet& iConfig):
  theTTRHBuilderName(iConfig.getParameter<std::string>("TTRHBuilder"))
{
  produces<PixelFitter>();
}

void TrackFitterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("TTRHBuilder", "");

  descriptions.add("trackFitter", desc);
}

void TrackFitterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::ESHandle<TrackerGeometry> trackerESH;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerESH);

  edm::ESHandle<MagneticField> fieldESH;
  iSetup.get<IdealMagneticFieldRecord>().get(fieldESH);

  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhbESH;
  iSetup.get<TransientRecHitRecord>().get(theTTRHBuilderName, ttrhbESH);

  auto impl = std::make_unique<TrackFitter>(&iSetup,
                                            trackerESH.product(),
                                            fieldESH.product(),
                                            ttrhbESH.product());
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(TrackFitterProducer);
