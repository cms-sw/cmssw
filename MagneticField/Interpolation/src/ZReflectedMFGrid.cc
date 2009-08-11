#include "MagneticField/Interpolation/src/ZReflectedMFGrid.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

#include <iostream>

using namespace std;

ZReflectedMFGrid::ZReflectedMFGrid( const GloballyPositioned<float>& vol,
				    MFGrid* sectorGrid) :
  MFGrid(vol), theSectorGrid( sectorGrid)

{}

ZReflectedMFGrid::~ZReflectedMFGrid()
{
  delete theSectorGrid;
}

MFGrid::LocalVector ZReflectedMFGrid::valueInTesla( const LocalPoint& p) const
{
  // Z reflection of point
  LocalPoint mirrorp( p.x(), p.y(), -p.z());
  LocalVector mirrorB = theSectorGrid->valueInTesla( mirrorp);
  return LocalVector( -mirrorB.x(), -mirrorB.y(), mirrorB.z());
}

void ZReflectedMFGrid::throwUp( const char* message) const
{
  std::cout << "Throwing exception " << message << std::endl;
  throw MagGeometryError(message);
}
void ZReflectedMFGrid::toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const 
{
  throwUp("Not implemented yet");
}

MFGrid::LocalPoint ZReflectedMFGrid::fromGridFrame( double a, double b, double c) const
{
  throwUp("Not implemented yet");
  return LocalPoint();
}

vector<int> ZReflectedMFGrid::dimensions() const {return theSectorGrid->dimensions();}

MFGrid::LocalPoint  ZReflectedMFGrid::nodePosition( int i, int j, int k) const
{
  throwUp("Not implemented yet");
  return LocalPoint();
}

MFGrid::LocalVector ZReflectedMFGrid::nodeValue( int i, int j, int k) const
{
  throwUp("Not implemented yet");
  return LocalVector();
}
