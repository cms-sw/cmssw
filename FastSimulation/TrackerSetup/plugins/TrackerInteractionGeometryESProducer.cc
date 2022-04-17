#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include <memory>
#include <string>

class TrackerInteractionGeometryESProducer : public edm::ESProducer {
public:
  TrackerInteractionGeometryESProducer(const edm::ParameterSet& p);
  ~TrackerInteractionGeometryESProducer() override = default;
  std::unique_ptr<TrackerInteractionGeometry> produce(const TrackerInteractionGeometryRecord&);

private:
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> geoSearchToken_;
  std::string label_;
  edm::ParameterSet theTrackerMaterial_;
};

TrackerInteractionGeometryESProducer::TrackerInteractionGeometryESProducer(const edm::ParameterSet& p) {
  auto cc = setWhatProduced(this);
  label_ = p.getUntrackedParameter<std::string>("trackerGeometryLabel", "");
  geoSearchToken_ = cc.consumes(edm::ESInputTag("", label_));
  theTrackerMaterial_ = p.getParameter<edm::ParameterSet>("TrackerMaterial");
}

std::unique_ptr<TrackerInteractionGeometry> TrackerInteractionGeometryESProducer::produce(
    const TrackerInteractionGeometryRecord& iRecord) {
  const GeometricSearchTracker* theGeomSearchTracker = &iRecord.get(geoSearchToken_);
  return std::make_unique<TrackerInteractionGeometry>(theTrackerMaterial_, theGeomSearchTracker);
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerInteractionGeometryESProducer);
