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
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class PixelFitterByConformalMappingAndLineProducer: public edm::global::EDProducer<> {
public:
  explicit PixelFitterByConformalMappingAndLineProducer(const edm::ParameterSet& iConfig):
    theTTRHBuilderName(iConfig.getParameter<std::string>("TTRHBuilder")),
    theFixImpactParameter(0), theUseFixImpactParameter(false)
  {
    if(iConfig.getParameter<bool>("useFixImpactParameter")) {
      theFixImpactParameter = iConfig.getParameter<double>("fixImpactParameter");
      theUseFixImpactParameter = true;
    }

    produces<PixelFitter>();
  }
  ~PixelFitterByConformalMappingAndLineProducer() {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("TTRHBuilder", "PixelTTRHBuilderWithoutAngle");
    desc.add<bool>("useFixImpactParameter", false);
    desc.add<double>("fixImpactParameter", 0.0);
    descriptions.add("pixelFitterByConformalMappingAndLine", desc);
  }

private:
  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  std::string theTTRHBuilderName;
  double theFixImpactParameter;
  bool theUseFixImpactParameter;
};


void PixelFitterByConformalMappingAndLineProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhBuilder;
  iSetup.get<TransientRecHitRecord>().get( theTTRHBuilderName, ttrhBuilder);

  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);

  edm::ESHandle<MagneticField> field;
  iSetup.get<IdealMagneticFieldRecord>().get(field);

  auto impl = std::make_unique<PixelFitterByConformalMappingAndLine>(&iSetup,
                                                                     ttrhBuilder.product(),
                                                                     tracker.product(),
                                                                     field.product(),
                                                                     theFixImpactParameter,
                                                                     theUseFixImpactParameter);
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(PixelFitterByConformalMappingAndLineProducer);
