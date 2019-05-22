#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include <memory>

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/ESProducer.h"

namespace edm {
  class ParameterSet;
}

class TransientRecHitRecord;

class MTDTransientTrackingRecHitBuilderESProducer : public edm::ESProducer {
public:
  /// Constructor
  MTDTransientTrackingRecHitBuilderESProducer(const edm::ParameterSet&);

  /// Destructor
  ~MTDTransientTrackingRecHitBuilderESProducer() override = default;

  // Operations
  std::unique_ptr<TransientTrackingRecHitBuilder> produce(const TransientRecHitRecord&);

protected:
private:
};

using namespace edm;
using namespace std;

MTDTransientTrackingRecHitBuilderESProducer::MTDTransientTrackingRecHitBuilderESProducer(
    const ParameterSet& parameterSet) {
  setWhatProduced(this, parameterSet.getParameter<string>("ComponentName"));
}

std::unique_ptr<TransientTrackingRecHitBuilder> MTDTransientTrackingRecHitBuilderESProducer::produce(
    const TransientRecHitRecord& iRecord) {
  ESHandle<GlobalTrackingGeometry> trackingGeometry;
  iRecord.getRecord<GlobalTrackingGeometryRecord>().get(trackingGeometry);

  return std::make_unique<MTDTransientTrackingRecHitBuilder>(trackingGeometry);
}

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Utilities/interface/typelookup.h"

DEFINE_FWK_EVENTSETUP_MODULE(MTDTransientTrackingRecHitBuilderESProducer);
