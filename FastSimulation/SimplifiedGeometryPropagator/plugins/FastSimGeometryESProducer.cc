#include "FastSimulation/SimplifiedGeometryPropagator/plugins/FastSimGeometryESProducer.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include <memory>

FastSimGeometryESProducer::FastSimGeometryESProducer(const edm::ParameterSet & p) 
{
    setWhatProduced(this);

    theTrackerMaterial = p.getParameter<edm::ParameterSet>("TrackerMaterial");
}

FastSimGeometryESProducer::~FastSimGeometryESProducer() {}

std::shared_ptr<fastsim::Geometry>
FastSimGeometryESProducer::produce(const GeometryRecord & iRecord){  
  _tracker = std::make_shared<fastsim::Geometry>(theTrackerMaterial);
  return _tracker;
}


DEFINE_FWK_EVENTSETUP_MODULE(FastSimGeometryESProducer);