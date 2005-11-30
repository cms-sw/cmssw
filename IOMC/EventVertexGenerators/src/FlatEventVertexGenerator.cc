#include "IOMC/EventVertexGenerators/interface/FlatEventVertexGenerator.h"
#include "Utilities/General/interface/CMSexception.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Random/RandFlat.h"

#include <iostream>

using std::cout;
using std::endl;

FlatEventVertexGenerator::FlatEventVertexGenerator(const edm::ParameterSet & p) 
: BaseEventVertexGenerator(p), m_pFlatEventVertexGenerator(p), myVertex(0)
{ 
  myMinX = m_pFlatEventVertexGenerator.getParameter<double>("MinX")*mm;
  myMinY = m_pFlatEventVertexGenerator.getParameter<double>("MinY")*mm;
  myMinZ = m_pFlatEventVertexGenerator.getParameter<double>("MinZ")*mm;
  myMaxX = m_pFlatEventVertexGenerator.getParameter<double>("MaxX")*mm;
  myMaxY = m_pFlatEventVertexGenerator.getParameter<double>("MaxY")*mm;
  myMaxZ = m_pFlatEventVertexGenerator.getParameter<double>("MaxZ")*mm;     

  if (myMinX > myMaxX) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in X - set to maximum in X = "
	 << myMaxX << " mm " << endl;
    myMinX = myMaxX; 
  }
  if (myMinY > myMaxY) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Y - set to maximum in Y = "
	 << myMaxY << " mm " << endl;
    myMinY = myMaxY; 
  }
  if (myMinZ > myMaxZ) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Z - set to maximum in Z = "
	 << myMaxZ << " mm " << endl;
    myMinZ = myMaxZ; 
  }
}

FlatEventVertexGenerator::~FlatEventVertexGenerator() 
{
  delete myVertex;
}

Hep3Vector * FlatEventVertexGenerator::newVertex() {
  delete myVertex;
  double aX,aY,aZ;
  aX = RandFlat::shoot(myMinX,myMaxX);
  aY = RandFlat::shoot(myMinY,myMaxY);
  aZ = RandFlat::shoot(myMinZ,myMaxZ);
  myVertex = new Hep3Vector(aX, aY, aZ);
  return myVertex;
}

Hep3Vector * FlatEventVertexGenerator::lastVertex() 
{
  return myVertex;
}

void FlatEventVertexGenerator::minX(double min) 
{
  if (min > myMaxX) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in X - value unchanged at X = "
	 << myMaxX << " mm " << endl;
  } else {
    myMinX = min;
  }
}

void FlatEventVertexGenerator::minY(double min) 
{
  if (min > myMaxY) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Y - value unchanged at Y = "
	 << myMaxY << " mm " << endl;
  } else {
    myMinY = min;
  }
}

void FlatEventVertexGenerator::minZ(double min) 
{
  if (min > myMaxZ) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal minimum in Z - value unchanged at Z = "
	 << myMaxZ << " mm " << endl;
  } else {
    myMinZ = min;
  }
}

void FlatEventVertexGenerator::maxX(double max) 
{
  if (max < myMinX) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal maximum in X - value unchanged at X = "
	 << myMaxX << " mm " << endl;
  } else {
    myMaxX = max;
  }
}

void FlatEventVertexGenerator::maxY(double max) 
{
  if (max < myMinY) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal maximum in Y - value unchanged at Y = "
	 << myMaxY << " mm " << endl;
  } else {
    myMaxY = max;
  }
}

void FlatEventVertexGenerator::maxZ(double max) 
{
  if (max < myMinZ) {
    cout << "Warning from FlatEventVertexGenerator: "
	 << "Illegal maximum in Z - value unchanged at Z = "
	 << myMaxZ << " mm " << endl;
  } else {
    myMaxZ = max;
  }
}

