#include "MagneticField/Interpolation/src/RectangularCylindricalMFGrid.h"
#include "MagneticField/Interpolation/src/binary_ifstream.h"
#include "MagneticField/Interpolation/src/LinearGridInterpolator3D.h"
#include <iostream>

using namespace std;

RectangularCylindricalMFGrid::RectangularCylindricalMFGrid( binary_ifstream& inFile, 
							    const GloballyPositioned<float>& vol)
  : MFGrid3D(vol)
{
  int n1, n2, n3;
  inFile >> n1 >> n2 >> n3;
  double xref, yref, zref;
  inFile >> xref >> yref >> zref;
  double stepx, stepy, stepz;
  inFile >> stepx    >> stepy    >> stepz;

  vector<BVector> fieldValues;
  float Bx, By, Bz;
  int nLines = n1*n2*n3;
  for (int iLine=0; iLine<nLines; ++iLine){
    inFile >> Bx >> By >> Bz;
    fieldValues.push_back(BVector(Bx,By,Bz));
  }
  // check completeness
  string lastEntry;
  inFile >> lastEntry;
  if (lastEntry != "complete"){
    cout << "error during file reading: file is not complete" << endl;
  }

  GlobalPoint grefp( GlobalPoint::Cylindrical( xref, yref, zref));
  LocalPoint lrefp = frame().toLocal( grefp);

#ifdef DEBUG_GRID
  cout << "Grid reference point in grid system: " << xref << "," << yref << "," << zref << endl;
  cout << "Grid reference point in global x,y,z: " << grefp << endl;
  cout << "Grid reference point in local x,y,z: " << lrefp << endl;
  cout << "steps " << stepx << "," <<  stepy << "," << stepz << endl;
#endif

  Grid1D<double> gridX( lrefp.perp(), lrefp.perp() + stepx*(n1-1), n1);
  //Grid1D<double> gridY( lrefp.phi(), lrefp.phi() + stepy*(n2-1), n2); // wrong: gives zero
  Grid1D<double> gridY( yref, yref + stepy*(n2-1), n2);
  Grid1D<double> gridZ( lrefp.z(), lrefp.z() + stepz*(n3-1), n3);

  grid_ = GridType( gridX, gridY, gridZ, fieldValues);
  
}

void RectangularCylindricalMFGrid::dump() const
{
  cout << endl << "Dump of RectangularCylindricalMFGrid" << endl;
  cout << "Number of points from Grid1D " 
       << grid_.grida().nodes() << " " << grid_.gridb().nodes() << " " << grid_.gridc().nodes() << endl;

  cout << "Reference Point from Grid1D " 
       << grid_.grida().lower() << " " << grid_.gridb().lower() << " " << grid_.gridc().lower() << endl;

  cout << "Basic Distance from Grid1D "
       << grid_.grida().step() << " " << grid_.gridb().step() << " " << grid_.gridc().step() << endl;


  cout << "Dumping " << grid_.data().size() << " field values " << endl;
  // grid_.dump();
}

MFGrid::LocalVector RectangularCylindricalMFGrid::uncheckedValueInTesla( const LocalPoint& p) const
{
//   static TimingReport::Item & timer= (*TimingReport::current())["MagneticFieldProvider::uncheckedValueInTesla(RectangularCylindricalMFGrid)"];
//   TimeMe t(timer,false);

  const float minimalSignificantR = 1e-6; // [cm], points below this radius are treated as zero radius
  LinearGridInterpolator3D<GridType::ValueType, GridType::Scalar> interpol( grid_);
  float R = p.perp();
  if (R < minimalSignificantR) {
    if (grid_.grida().lower() < minimalSignificantR) {
      int k = grid_.gridc().index(p.z());
      double u = (p.z() - grid_.gridc().node(k)) / grid_.gridc().step();
      LocalVector result((1-u)*grid_(0,  0,  k) + u*grid_(0,  0,  k+1)); 
      return result;
    }
  }

  // FIXME: "OLD" convention of phi.
  // GridType::ValueType value = interpol( R, Geom::pi() - p.phi(), p.z());
  GridType::ValueType value = interpol( R, p.phi(), p.z());
  return LocalVector(value);
}

void RectangularCylindricalMFGrid::toGridFrame( const LocalPoint& p, 
					      double& a, double& b, double& c) const
{
  a = p.perp();
  // FIXME: "OLD" convention of phi.
  //  b = Geom::pi() - p.phi();
  b = p.phi();
  c = p.z();
}
 
MFGrid::LocalPoint RectangularCylindricalMFGrid::fromGridFrame( double a, double b, double c) const
{

  // FIXME: "OLD" convention of phi.
  //  return LocalPoint( LocalPoint::Cylindrical(a, Geom::pi() - b, c));
  return LocalPoint( LocalPoint::Cylindrical(a, b, c));
}
