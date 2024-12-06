
/*
*/

#include "IOMC/EventVertexGenerators/interface/PassThroughEvtVtxGenerator.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace edm;

PassThroughEvtVtxGenerator::PassThroughEvtVtxGenerator(const ParameterSet& pset) : BaseEvtVtxGenerator(pset) {
  Service<RandomNumberGenerator> rng;
}

PassThroughEvtVtxGenerator::~PassThroughEvtVtxGenerator() {}

ROOT::Math::XYZTVector PassThroughEvtVtxGenerator::vertexShift(CLHEP::HepRandomEngine*) const {
  return ROOT::Math::XYZTVector(0., 0., 0., 0.);
}
