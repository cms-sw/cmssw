#include "FastSimulation/TrackerSetup/plugins/TrackerInteractionGeometryESProducer.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include <memory>
#include <string>

using namespace edm;

TrackerInteractionGeometryESProducer::TrackerInteractionGeometryESProducer(const edm::ParameterSet & p) 
{
    setWhatProduced(this);
    _label = p.getUntrackedParameter<std::string>("trackerGeometryLabel","");

    theTrackerMaterial = p.getParameter<edm::ParameterSet>("TrackerMaterial");

}

TrackerInteractionGeometryESProducer::~TrackerInteractionGeometryESProducer() {}

boost::shared_ptr<TrackerInteractionGeometry> 
TrackerInteractionGeometryESProducer::produce(const TrackerInteractionGeometryRecord & iRecord){ 

  edm::ESHandle<GeometricSearchTracker> theGeomSearchTracker;
  
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(_label, theGeomSearchTracker );
  _tracker = boost::shared_ptr<TrackerInteractionGeometry>
    (new TrackerInteractionGeometry(theTrackerMaterial,&(*theGeomSearchTracker)));
  return _tracker;

}


DEFINE_FWK_EVENTSETUP_MODULE(TrackerInteractionGeometryESProducer);
