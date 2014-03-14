

#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
//#include "CLHEP/Vector/ThreeVector.h"
#include "HepMC/SimpleVector.h"

FlatEvtVtxGenerator::FlatEvtVtxGenerator(const edm::ParameterSet& p )
: BaseEvtVtxGenerator(p)
{
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

FlatEvtVtxGenerator::~FlatEvtVtxGenerator()
{
}

//Hep3Vector * FlatEvtVtxGenerator::newVertex() {
HepMC::FourVector* FlatEvtVtxGenerator::newVertex(CLHEP::HepRandomEngine* engine) {
  double aX,aY,aZ;
  aX = CLHEP::RandFlat::shoot(engine, fMinX, fMaxX);
  aY = CLHEP::RandFlat::shoot(engine, fMinY, fMaxY);
  aZ = CLHEP::RandFlat::shoot(engine, fMinZ, fMaxZ);

  //if (fVertex == 0) fVertex = new CLHEP::Hep3Vector;
  //fVertex->set(aX,aY,aZ);
  if ( fVertex == 0 ) fVertex = new HepMC::FourVector() ;
  fVertex->set(aX,aY,aZ,fTimeOffset);

  return fVertex;
}

void FlatEvtVtxGenerator::minX(double min) 
{
  fMinX = min;
}

void FlatEvtVtxGenerator::minY(double min) 
{
  fMinY = min;
}

void FlatEvtVtxGenerator::minZ(double min) 
{
  fMinZ = min;
}

void FlatEvtVtxGenerator::maxX(double max) 
{
  fMaxX = max;
}

void FlatEvtVtxGenerator::maxY(double max) 
{
  fMaxY = max;
}

void FlatEvtVtxGenerator::maxZ(double max) 
{
  fMaxZ = max;
}
