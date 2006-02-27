#include "RecoTracker/GeometryESProducer/interface/TrackerRecoGeometryESProducer.h"
#include "Geometry/TrackerBaseAlgo/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTrackerBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"


#include <memory>

using namespace edm;

TrackerRecoGeometryESProducer::TrackerRecoGeometryESProducer(const edm::ParameterSet & p) 
{
    setWhatProduced(this);
}

TrackerRecoGeometryESProducer::~TrackerRecoGeometryESProducer() {}

boost::shared_ptr<GeometricSearchTracker> 
TrackerRecoGeometryESProducer::produce(const TrackerRecoGeometryRecord & iRecord){ 
  //
  // get the DDCompactView first
  //
  edm::ESHandle<GeometricDet> gD;
  edm::ESHandle<TrackingGeometry> tG;
  iRecord.getRecord<IdealGeometryRecord>().get( gD );
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(tG );
  GeometricSearchTrackerBuilder builder;
  _tracker  = boost::shared_ptr<GeometricSearchTracker>(builder.build( &(*gD), &(*tG) ));
  return _tracker;
}


DEFINE_FWK_EVENTSETUP_MODULE(TrackerRecoGeometryESProducer)
