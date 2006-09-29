#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"
#include "Utilities/General/interface/CMSexception.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Random/RandFlat.h"

#include <iostream>

using std::cout;
using std::endl;

using namespace edm;

FlatEvtVtxGenerator::FlatEvtVtxGenerator(const edm::ParameterSet& p )
: BaseEvtVtxGenerator(p)
{ 
  
  fRandom = new RandFlat(fEngine) ;
  
  fMinX = p.getParameter<double>("MinX")*cm;
  fMinY = p.getParameter<double>("MinY")*cm;
  fMinZ = p.getParameter<double>("MinZ")*cm;
  fMaxX = p.getParameter<double>("MaxX")*cm;
  fMaxY = p.getParameter<double>("MaxY")*cm;
  fMaxZ = p.getParameter<double>("MaxZ")*cm;     

  if (fMinX > fMaxX) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in X - set to maximum in X = "
	 << fMaxX << " mm " << endl;
    fMinX = fMaxX; 
  }
  if (fMinY > fMaxY) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Y - set to maximum in Y = "
	 << fMaxY << " mm " << endl;
    fMinY = fMaxY; 
  }
  if (fMinZ > fMaxZ) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Z - set to maximum in Z = "
	 << fMaxZ << " mm " << endl;
    fMinZ = fMaxZ; 
  }
}

FlatEvtVtxGenerator::~FlatEvtVtxGenerator() 
{
  // I'm not deleting this, since the engine seems to have
  // been delete earlier; thus an attempt tp delete RandFlat
  // results in a core dump... 
  // I need to ask Marc/Jim how to do it right...
  //delete fRandom; 
}

Hep3Vector * FlatEvtVtxGenerator::newVertex() {
  if ( fVertex != NULL ) delete fVertex;
  double aX,aY,aZ;
  aX = fRandom->fire(fMinX,fMaxX) ;
  aY = fRandom->fire(fMinY,fMaxY) ;
  aZ = fRandom->fire(fMinZ,fMaxZ) ;
  fVertex = new Hep3Vector(aX, aY, aZ);
  return fVertex;
}

void FlatEvtVtxGenerator::minX(double min) 
{
  if (min > fMaxX) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in X - value unchanged at X = "
	 << fMaxX << " mm " << endl;
  } else {
    fMinX = min;
  }
}

void FlatEvtVtxGenerator::minY(double min) 
{
  if (min > fMaxY) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Y - value unchanged at Y = "
	 << fMaxY << " mm " << endl;
  } else {
    fMinY = min;
  }
}

void FlatEvtVtxGenerator::minZ(double min) 
{
  if (min > fMaxZ) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Z - value unchanged at Z = "
	 << fMaxZ << " mm " << endl;
  } else {
    fMinZ = min;
  }
}

void FlatEvtVtxGenerator::maxX(double max) 
{
  if (max < fMinX) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal maximum in X - value unchanged at X = "
	 << fMaxX << " mm " << endl;
  } else {
    fMaxX = max;
  }
}

void FlatEvtVtxGenerator::maxY(double max) 
{
  if (max < fMinY) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal maximum in Y - value unchanged at Y = "
	 << fMaxY << " mm " << endl;
  } else {
    fMaxY = max;
  }
}

void FlatEvtVtxGenerator::maxZ(double max) 
{
  if (max < fMinZ) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal maximum in Z - value unchanged at Z = "
	 << fMaxZ << " mm " << endl;
  } else {
    fMaxZ = max;
  }
}

