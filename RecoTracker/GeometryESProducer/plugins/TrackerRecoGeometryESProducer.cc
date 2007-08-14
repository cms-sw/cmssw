#include "RecoTracker/GeometryESProducer/plugins/TrackerRecoGeometryESProducer.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
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
  edm::ESHandle<TrackerGeometry> tG;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(tG );
  GeometricSearchTrackerBuilder builder;
  _tracker  = boost::shared_ptr<GeometricSearchTracker>(builder.build( tG->trackerDet(), &(*tG) ));
  return _tracker;
}


DEFINE_FWK_EVENTSETUP_MODULE(TrackerRecoGeometryESProducer);
