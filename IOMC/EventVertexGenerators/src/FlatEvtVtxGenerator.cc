

#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "CLHEP/Vector/ThreeVector.h"
#include "HepMC/SimpleVector.h"

FlatEvtVtxGenerator::FlatEvtVtxGenerator(const edm::ParameterSet& p) : BaseEvtVtxGenerator(p) {
  fMinX = p.getParameter<double>("MinX") * cm;
  fMinY = p.getParameter<double>("MinY") * cm;
  fMinZ = p.getParameter<double>("MinZ") * cm;
  fMaxX = p.getParameter<double>("MaxX") * cm;
  fMaxY = p.getParameter<double>("MaxY") * cm;
  fMaxZ = p.getParameter<double>("MaxZ") * cm;
  fMinT = p.getParameter<double>("MinT") * ns * c_light;
  fMaxT = p.getParameter<double>("MaxT") * ns * c_light;

  if (fMinX > fMaxX) {
    throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                          << "MinX is greater than MaxX";
  }
  if (fMinY > fMaxY) {
    throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                          << "MinY is greater than MaxY";
  }
  if (fMinZ > fMaxZ) {
    throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                          << "MinZ is greater than MaxZ";
  }
  if (fMinT > fMaxT) {
    throw cms::Exception("Configuration") << "Error in FlatEvtVtxGenerator: "
                                          << "MinT is greater than MaxT";
  }
  edm::LogVerbatim("FlatEvtVtx") << "FlatEvtVtxGenerator Initialized with x[" << fMinX << ":" << fMaxX << "] cm; y[" << fMinY << ":" << fMaxY << "] cm; z[" << fMinZ << ":" << fMaxZ << "] cm; t[" << fMinT << ":" << fMaxT << "]";
}

FlatEvtVtxGenerator::~FlatEvtVtxGenerator() {}

//Hep3Vector * FlatEvtVtxGenerator::newVertex() {
HepMC::FourVector FlatEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine* engine) const {
  double aX, aY, aZ, aT;
  aX = CLHEP::RandFlat::shoot(engine, fMinX, fMaxX);
  aY = CLHEP::RandFlat::shoot(engine, fMinY, fMaxY);
  aZ = CLHEP::RandFlat::shoot(engine, fMinZ, fMaxZ);
  aT = CLHEP::RandFlat::shoot(engine, fMinT, fMaxT);

  edm::LogVerbatim("FlatEvtVtx") << "FlatEvtVtxGenerator Vertex at [" << aX <<", " << aY << ", " << aZ << ", " << aT << "]";

  return HepMC::FourVector(aX, aY, aZ, aT);
}

void FlatEvtVtxGenerator::minX(double min) { fMinX = min; }

void FlatEvtVtxGenerator::minY(double min) { fMinY = min; }

void FlatEvtVtxGenerator::minZ(double min) { fMinZ = min; }

void FlatEvtVtxGenerator::maxX(double max) { fMaxX = max; }

void FlatEvtVtxGenerator::maxY(double max) { fMaxY = max; }

void FlatEvtVtxGenerator::maxZ(double max) { fMaxZ = max; }
