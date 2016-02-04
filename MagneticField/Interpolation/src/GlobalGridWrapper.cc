#include "GlobalGridWrapper.h"
#include "MagneticFieldGrid.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

using namespace std;

GlobalGridWrapper::GlobalGridWrapper( const GloballyPositioned<float>& vol,
				      const string& fileName)
  : MFGrid(vol)
{
  theRealOne = new MagneticFieldGrid;
  theRealOne->load(fileName);
}

MFGrid::LocalVector GlobalGridWrapper::valueInTesla( const LocalPoint& p) const
{

  GlobalPoint gp = frame().toGlobal(p);
  float bx, by, bz;

  int gridType = theRealOne->gridType();
  if ( gridType == 1 || gridType == 2) {
    // x,y,z grid
    theRealOne->interpolateAtPoint( gp.x(), gp.y(), gp.z(), bx, by, bz);      
  }
  else {
    // r,phi,z grid
//     cout << "calling interpolateAtPoint with args " 
// 	 << gp.perp() << " " << gp.phi() << " " << gp.z() << endl;
    theRealOne->interpolateAtPoint( gp.perp(), gp.phi(), gp.z(), bx, by, bz);      
//     cout << "interpolateAtPoint returned " 
// 	 << bx << " " << by << " " << bz << endl;
  }
  return LocalVector( bx, by, bz);
}

void GlobalGridWrapper::dump() const {}

void GlobalGridWrapper::toGridFrame( const LocalPoint& p, 
					      double& a, double& b, double& c) const
{
  throw MagLogicError ("GlobalGridWrapper::toGridFrame not implemented yet");
}
 
MFGrid::LocalPoint GlobalGridWrapper::fromGridFrame( double a, double b, double c) const
{
  throw MagLogicError ("GlobalGridWrapper::fromGridFrame not implemented yet");
  return LocalPoint( 0, 0, 0);
}

Dimensions GlobalGridWrapper::dimensions() const
{
  throw MagLogicError ("GlobalGridWrapper::dimensions not implemented yet");
  return Dimensions();
}

MFGrid::LocalPoint GlobalGridWrapper::nodePosition( int i, int j, int k) const
{
  throw MagLogicError ("GlobalGridWrapper::nodePosition not implemented yet");
  return LocalPoint( 0, 0, 0);
}

MFGrid::LocalVector GlobalGridWrapper::nodeValue( int i, int j, int k) const
{
  throw MagLogicError ("GlobalGridWrapper::nodeValue not implemented yet");
  return LocalVector( 0, 0, 0);
}

