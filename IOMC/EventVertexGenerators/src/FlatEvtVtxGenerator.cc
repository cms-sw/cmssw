

#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "HepMC/SimpleVector.h"

FlatEvtVtxGenerator::FlatEvtVtxGenerator(edm::ParameterSet const& p, edm::ConsumesCollector&) :
    BaseEvtVtxGenerator(), fVertex(new HepMC::FourVector()) {
  fMinX = p.getParameter<double>("MinX")*cm;
  fMinY = p.getParameter<double>("MinY")*cm;
  fMinZ = p.getParameter<double>("MinZ")*cm;
  fMaxX = p.getParameter<double>("MaxX")*cm;
  fMaxY = p.getParameter<double>("MaxY")*cm;
  fMaxZ = p.getParameter<double>("MaxZ")*cm;
  fTimeOffset = p.getParameter<double>("TimeOffset")*ns*c_light;

  if (fMinX > fMaxX) {
    throw cms::Exception("Configuration")
      << "Error in FlatEvtVtxGenerator: "
      << "MinX is greater than MaxX";
  }
  if (fMinY > fMaxY) {
    throw cms::Exception("Configuration")
      << "Error in FlatEvtVtxGenerator: "
      << "MinY is greater than MaxY";
  }
  if (fMinZ > fMaxZ) {
    throw cms::Exception("Configuration")
      << "Error in FlatEvtVtxGenerator: "
      << "MinZ is greater than MaxZ";
  }
}

FlatEvtVtxGenerator::~FlatEvtVtxGenerator() {
}

void FlatEvtVtxGenerator::generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) {
   product.applyVtxGen(newVertex(engine));
}

HepMC::FourVector* FlatEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine& engine) {
  double aX = CLHEP::RandFlat::shoot(&engine, fMinX, fMaxX);
  double aY = CLHEP::RandFlat::shoot(&engine, fMinY, fMaxY);
  double aZ = CLHEP::RandFlat::shoot(&engine, fMinZ, fMaxZ);
  double aT = CLHEP::RandFlat::shoot(&engine, fMinZ, fMaxZ);

  fVertex->set(aX,aY,aZ,aT+fTimeOffset);
  return fVertex.get();
}

void FlatEvtVtxGenerator::minX(double min) {
  fMinX = min;
}

void FlatEvtVtxGenerator::minY(double min) {
  fMinY = min;
}

void FlatEvtVtxGenerator::minZ(double min) {
  fMinZ = min;
}

void FlatEvtVtxGenerator::maxX(double max) {
  fMaxX = max;
}

void FlatEvtVtxGenerator::maxY(double max) {
  fMaxY = max;
}

void FlatEvtVtxGenerator::maxZ(double max) {
  fMaxZ = max;
}
