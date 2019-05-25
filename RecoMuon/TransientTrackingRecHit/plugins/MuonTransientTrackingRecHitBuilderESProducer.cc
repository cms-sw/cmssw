#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoMuon/TransientTrackingRecHit/plugins/MuonTransientTrackingRecHitBuilderESProducer.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <memory>

using namespace edm;
using namespace std;

MuonTransientTrackingRecHitBuilderESProducer::MuonTransientTrackingRecHitBuilderESProducer(
    const ParameterSet& parameterSet) {
  setWhatProduced(this, parameterSet.getParameter<string>("ComponentName"));
}

MuonTransientTrackingRecHitBuilderESProducer::~MuonTransientTrackingRecHitBuilderESProducer() {}

std::unique_ptr<TransientTrackingRecHitBuilder> MuonTransientTrackingRecHitBuilderESProducer::produce(
    const TransientRecHitRecord& iRecord) {
  ESHandle<GlobalTrackingGeometry> trackingGeometry;
  iRecord.getRecord<GlobalTrackingGeometryRecord>().get(trackingGeometry);

  return std::make_unique<MuonTransientTrackingRecHitBuilder>(trackingGeometry);
}
