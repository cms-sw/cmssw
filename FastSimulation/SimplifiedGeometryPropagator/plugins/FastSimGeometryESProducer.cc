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

std::unique_ptr<fastsim::Geometry>
FastSimGeometryESProducer::produce(const GeometryRecord & iRecord){
    return std::make_unique<fastsim::Geometry>(theTrackerMaterial);
}

DEFINE_FWK_EVENTSETUP_MODULE(FastSimGeometryESProducer);
