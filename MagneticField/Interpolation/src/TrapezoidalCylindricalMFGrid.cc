#include "MagneticField/Interpolation/interface/TrapezoidalCylindricalMFGrid.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "MagneticField/Interpolation/interface/LinearGridInterpolator3D.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

// #include "Utilities/Notification/interface/TimingReport.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"

#include <iostream>

using namespace std;

TrapezoidalCylindricalMFGrid::TrapezoidalCylindricalMFGrid( binary_ifstream& inFile,
							const GloballyPositioned<float>& vol)
  : MFGrid3D(vol)
{
  int n1, n2, n3;
  inFile >> n1 >> n2 >> n3;
  double xref, yref, zref;
  inFile >> xref >> yref >> zref;
  double stepx, stepy, stepz;
  inFile >> stepx    >> stepy    >> stepz;

  double BasicDistance1[3][3];  // linear step
  double BasicDistance2[3][3];  // linear offset
  bool   easya, easyb, easyc;

  inFile >> BasicDistance1[0][0] >> BasicDistance1[1][0] >> BasicDistance1[2][0];
  inFile >> BasicDistance1[0][1] >> BasicDistance1[1][1] >> BasicDistance1[2][1];
  inFile >> BasicDistance1[0][2] >> BasicDistance1[1][2] >> BasicDistance1[2][2];
  inFile >> BasicDistance2[0][0] >> BasicDistance2[1][0] >> BasicDistance2[2][0];
  inFile >> BasicDistance2[0][1] >> BasicDistance2[1][1] >> BasicDistance2[2][1];
  inFile >> BasicDistance2[0][2] >> BasicDistance2[1][2] >> BasicDistance2[2][2];
  inFile >> easya >> easyb >> easyc;

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
  if (lastEntry != "complete") {
    cout << "error during file reading: file is not complete" << endl;
  }

#ifdef DEBUG_GRID
  cout << "easya " << easya << " easyb " << easyb << " easyc " << easyc << endl;
#endif

  if (!easyb || !easyc) {
    throw MagGeometryError("TrapezoidalCartesianMFGrid only implemented for first coordinate");
  }

#ifdef DEBUG_GRID
  cout << "Grid reference point in grid system: " << xref << "," << yref << "," << zref << endl;
  cout << "steps " << stepx << "," <<  stepy << "," << stepz << endl;
  cout << "ns " << n1 << "," <<  n2 << "," << n3 << endl;

  for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
    cout << "BasicDistance1[" << i << "][" << j << "] = " << BasicDistance1[i][j]
	 << "BasicDistance2[" << i << "][" << j << "] = " << BasicDistance2[i][j] << endl;
  }
#endif

  // the "not easy" coordinate is x
  double a = stepx * (n1 -1);
  double b = a + BasicDistance1[0][1] * (n2-1)*(n1-1) + BasicDistance1[0][2] * (n3-1)*(n1-1);
  //  double h = stepy * (n2-1);
  double h = stepz * (n3-1);
  double delta = -BasicDistance2[0][1] * (n2-1) -BasicDistance2[0][2] * (n3-1);

#ifdef DEBUG_GRID
  cout << "Trapeze size (a,b,h) = " << a << "," << b << "," << h << endl;
#endif

  GlobalPoint grefp( GlobalPoint::Cylindrical( xref, Geom::pi() - yref, zref));
  LocalPoint lrefp = frame().toLocal( grefp);

#ifdef DEBUG_GRID
  cout << "Global origin " << grefp << endl;
  cout << "Local origin  " << lrefp << endl;
#endif

  double baMinus1 = BasicDistance1[0][2]*(n3-1) / stepx;
  if (abs(baMinus1) > 0.000001) {
    double b_over_a = 1 + baMinus1;
    double a1 = abs(baMinus1) > 0.000001 ? delta / baMinus1 : a/2;
#ifdef DEBUG_GRID
   cout << "a1 = " << a1 << endl;
#endif

    // transform reference point to grid frame
    double x0 = lrefp.perp() + a1;
    double y0 = lrefp.z() + h/2.;
    mapping_ = Trapezoid2RectangleMappingX( x0, y0, b_over_a, h);
  }
  else { // parallelogram
    mapping_ = Trapezoid2RectangleMappingX( 0, 0, delta/h);
  }
  double xrec, yrec;
  mapping_.rectangle( lrefp.perp(), lrefp.z(), xrec, yrec);

  Grid1D<double> gridX( xrec, xrec + (a+b)/2., n1);
  Grid1D<double> gridY( yref, yref + stepy*(n2-1), n2);
  Grid1D<double> gridZ( yrec, yrec + h, n3);
  grid_ = GridType( gridX, gridY, gridZ, fieldValues);
    
  // Activate/deactivate timers
//   static SimpleConfigurable<bool> timerOn(false,"MFGrid:timing");
//   (*TimingReport::current()).switchOn("MagneticFieldProvider::valueInTesla(TrapezoidalCylindricalMFGrid)",timerOn);
}

void TrapezoidalCylindricalMFGrid::dump() const
{}

MFGrid::LocalVector 
TrapezoidalCylindricalMFGrid::uncheckedValueInTesla( const LocalPoint& p) const
{
//   static TimingReport::Item & timer= (*TimingReport::current())["MagneticFieldProvider::valueInTesla(TrapezoidalCylindricalMFGrid)"];
//   TimeMe t(timer,false);

  LinearGridInterpolator3D<GridType::ValueType, GridType::Scalar> interpol( grid_);
  double a, b, c;
  toGridFrame( p, a, b, c);
  GlobalVector gv( interpol( a, b, c)); // grid in global frame
  return frame().toLocal(gv);           // must return a local vector
}

void TrapezoidalCylindricalMFGrid::toGridFrame( const LocalPoint& p, 
					      double& a, double& b, double& c) const
{
  mapping_.rectangle( p.perp(), p.z(), a, c);
  b = Geom::pi() - p.phi();
}

MFGrid::LocalPoint 
TrapezoidalCylindricalMFGrid::fromGridFrame( double a, double b, double c) const
{
  double rtrap, ztrap;
  mapping_.trapezoid( a, c, rtrap, ztrap);
  return LocalPoint(LocalPoint::Cylindrical(rtrap, Geom::pi() - b, ztrap));
}
