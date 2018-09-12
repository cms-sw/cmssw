#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoMTD/TransientTrackingRecHit/plugins/MTDTransientTrackingRecHitBuilderESProducer.h"
#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include<memory>

using namespace edm;
using namespace std;
    
MTDTransientTrackingRecHitBuilderESProducer::MTDTransientTrackingRecHitBuilderESProducer(const ParameterSet & parameterSet) {

  setWhatProduced(this,parameterSet.getParameter<string>("ComponentName"));
}
    
MTDTransientTrackingRecHitBuilderESProducer::~MTDTransientTrackingRecHitBuilderESProducer() {}

    
std::unique_ptr<TransientTrackingRecHitBuilder> 
MTDTransientTrackingRecHitBuilderESProducer::produce(const TransientRecHitRecord& iRecord){ 
  

  ESHandle<GlobalTrackingGeometry> trackingGeometry;
  iRecord.getRecord<GlobalTrackingGeometryRecord>().get(trackingGeometry);
  
  return std::make_unique<MTDTransientTrackingRecHitBuilder>(trackingGeometry);
}
    
    
