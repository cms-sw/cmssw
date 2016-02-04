#include "CylinderFromSectorMFGrid.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"
#include <iostream>

CylinderFromSectorMFGrid::CylinderFromSectorMFGrid( const GloballyPositioned<float>& vol,
						    double phiMin, double phiMax,
						    MFGrid* sectorGrid) :
  MFGrid(vol), thePhiMin(phiMin), thePhiMax(phiMax), theSectorGrid( sectorGrid)

{
   if (thePhiMax < thePhiMin) thePhiMax += 2.0*Geom::pi();
   theDelta = thePhiMax - thePhiMin;
}

CylinderFromSectorMFGrid::~CylinderFromSectorMFGrid()
{
  delete theSectorGrid;
}

MFGrid::LocalVector CylinderFromSectorMFGrid::valueInTesla( const LocalPoint& p) const
{
  double phi = p.phi();
  if (phi < thePhiMax && phi > thePhiMin) return theSectorGrid->valueInTesla(p);
  else {
    double phiRot = floor((phi-thePhiMin)/theDelta) * theDelta;
    double c = cos(phiRot);
    double s = sin(phiRot);
    double xrot =  p.x()*c + p.y()*s;
    double yrot = -p.x()*s + p.y()*c;

    // get field in interpolation sector 
    MFGrid::LocalVector tmp = theSectorGrid->valueInTesla( LocalPoint(xrot,yrot,p.z()));

    // rotate field back to original sector
    return MFGrid::LocalVector( tmp.x()*c - tmp.y()*s, tmp.x()*s + tmp.y()*c, tmp.z());
  }
}

void CylinderFromSectorMFGrid::throwUp( const char *message) const
{
  std::cout << "Throwing exception " << message << std::endl;
  throw MagGeometryError(message);
}
void CylinderFromSectorMFGrid::toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const 
{
  throwUp("Not implemented yet");
}

MFGrid::LocalPoint CylinderFromSectorMFGrid::fromGridFrame( double a, double b, double c) const
{
  throwUp("Not implemented yet");
  return LocalPoint();
}

Dimensions CylinderFromSectorMFGrid::dimensions() const 
{return theSectorGrid->dimensions();}

MFGrid::LocalPoint  CylinderFromSectorMFGrid::nodePosition( int i, int j, int k) const
{
  throwUp("Not implemented yet");
  return LocalPoint();
}

MFGrid::LocalVector CylinderFromSectorMFGrid::nodeValue( int i, int j, int k) const
{
  throwUp("Not implemented yet");
  return LocalVector();
}
