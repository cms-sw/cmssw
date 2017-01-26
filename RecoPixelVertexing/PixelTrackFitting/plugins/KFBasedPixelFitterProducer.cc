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
#include "RecoPixelVertexing/PixelTrackFitting/interface/KFBasedPixelFitter.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class KFBasedPixelFitterProducer: public edm::global::EDProducer<> {
public:
  explicit KFBasedPixelFitterProducer(const edm::ParameterSet& iConfig);
  ~KFBasedPixelFitterProducer() {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  std::string thePropagatorLabel;
  std::string thePropagatorOppositeLabel;
  std::string theTTRHBuilderName;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;
};

KFBasedPixelFitterProducer::KFBasedPixelFitterProducer(const edm::ParameterSet& iConfig):
  thePropagatorLabel(iConfig.getParameter<std::string>("propagator")),
  thePropagatorOppositeLabel(iConfig.getParameter<std::string>("propagator")),
  theTTRHBuilderName(iConfig.getParameter<std::string>("TTRHBuilder"))
{
  if(iConfig.getParameter<bool>("useBeamSpotConstraint")) {
    theBeamSpotToken = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotConstraint"));
  }

  produces<PixelFitter>();
}

void KFBasedPixelFitterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("useBeamSpotConstraint", true);
  desc.add<edm::InputTag>("beamSpotConstraint", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<std::string>("propagatorOpposite", "PropagatorWithMaterialOpposite");
  desc.add<std::string>("TTRHBuilder", "PixelTTRHBuilderWithoutAngle");

  descriptions.add("kfBasedPixelFitter", desc);
}

void KFBasedPixelFitterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhb;
  iSetup.get<TransientRecHitRecord>().get( theTTRHBuilderName, ttrhb);

  edm::ESHandle<Propagator>  propagator;
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorLabel, propagator);

  edm::ESHandle<Propagator>  opropagator;
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorOppositeLabel, opropagator);

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  edm::ESHandle<MagneticField> field;
  iSetup.get<IdealMagneticFieldRecord>().get(field);

  const reco::BeamSpot *beamspot = nullptr;
  if(!theBeamSpotToken.isUninitialized()) {
    edm::Handle<reco::BeamSpot> hbs;
    iEvent.getByToken(theBeamSpotToken, hbs);
    beamspot = hbs.product();
  }

  auto impl = std::make_unique<KFBasedPixelFitter>(&iSetup,
                                                   propagator.product(),
                                                   opropagator.product(),
                                                   ttrhb.product(),
                                                   tracker.product(),
                                                   field.product(),
                                                   beamspot);
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(KFBasedPixelFitterProducer);
