#ifndef FastSimulation_SimplifiedGeometryPropagator_GeometryESProducer_H
#define FastSimulation_SimplifiedGeometryPropagator_GeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/GeometryRecord.h"
#include "FastSimulation/SimplifiedGeometryPropagator/interface/Geometry.h"
#include <memory>
#include <string>

class FastSimGeometryESProducer: public edm::ESProducer{
    public:
    FastSimGeometryESProducer(const edm::ParameterSet & p);
    ~FastSimGeometryESProducer() override; 
    std::unique_ptr<fastsim::Geometry> produce(const GeometryRecord &);
    private:
    edm::ParameterSet theTrackerMaterial;
};


#endif



