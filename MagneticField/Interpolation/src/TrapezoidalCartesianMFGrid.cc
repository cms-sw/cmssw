#include "MagneticField/Interpolation/interface/TrapezoidalCartesianMFGrid.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "MagneticField/Interpolation/interface/LinearGridInterpolator3D.h"
#include "MagneticField/VolumeGeometry/interface/MagExceptions.h"

// #include "Utilities/Notification/interface/TimingReport.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"

#include <iostream>

using namespace std;

TrapezoidalCartesianMFGrid::TrapezoidalCartesianMFGrid( binary_ifstream& inFile,
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

  if (!easyb || !easyc) {
    throw MagGeometryError("TrapezoidalCartesianMFGrid only implemented for first coordinate");
  }

  // the "not easy" coordinate is x
  double a = stepx * (n1 -1);
  double b = a + BasicDistance1[0][1] * (n2-1)*(n1-1);
  double h = stepy * (n2-1);
  double delta = -BasicDistance2[0][1] * (n2-1);

#ifdef DEBUG_GRID
  cout << "Trapeze size (a,b,h) = " << a << "," << b << "," << h << endl;
#endif

  GlobalPoint grefp( xref, yref, zref);
  LocalPoint lrefp = frame().toLocal( grefp);

#ifdef DEBUG_GRID
  cout << "Global origin " << grefp << endl;
  cout << "Local origin  " << lrefp << endl;
#endif

  double baMinus1 = BasicDistance1[0][1]*(n2-1) / stepx;
  if (abs(baMinus1) > 0.000001) {
    double b_over_a = 1 + baMinus1;
    double a1 = abs(baMinus1) > 0.000001 ? delta / baMinus1 : a/2;

#ifdef DEBUG_GRID
    cout << "a1 = " << a1 << endl;
#endif

    // transform reference point to grid frame
    double x0 = lrefp.x() + a1;
    double y0 = lrefp.y() + h/2.;
    mapping_ = Trapezoid2RectangleMappingX( x0, y0, b_over_a, h);
  }
  else { // parallelogram
    mapping_ = Trapezoid2RectangleMappingX( 0, 0, delta/h);
  }
  double xrec, yrec;
  mapping_.rectangle( lrefp.x(), lrefp.y(), xrec, yrec);

  Grid1D<double> gridX( xrec, xrec + (a+b)/2., n1);
  Grid1D<double> gridY( yrec, yrec + h, n2);
  Grid1D<double> gridZ( lrefp.z(), lrefp.z() + stepz*(n3-1), n3);
  grid_ = GridType( gridX, gridY, gridZ, fieldValues);
    
  // Activate/deactivate timers
//   static SimpleConfigurable<bool> timerOn(false,"MFGrid:timing");
//   (*TimingReport::current()).switchOn("MagneticFieldProvider::valueInTesla(TrapezoidalCartesianMFGrid)",timerOn);

}

void TrapezoidalCartesianMFGrid::dump() const
{
  
  cout << endl << "Dump of TrapezoidalCartesianMFGrid" << endl;
//   cout << "Number of points from file   " 
//        << n1 << " " << n2 << " " << n3 << endl;
  cout << "Number of points from Grid1D " 
       << grid_.grida().nodes() << " " << grid_.gridb().nodes() << " " << grid_.gridc().nodes() << endl;

//   cout << "Reference Point from file   " 
//        << xref << " " << yref << " " << zref << endl;
  cout << "Reference Point from Grid1D " 
       << grid_.grida().lower() << " " << grid_.gridb().lower() << " " << grid_.gridc().lower() << endl;

//   cout << "Basic Distance from file   " 
//        <<  stepx << " " << stepy << " " << stepz << endl;
  cout << "Basic Distance from Grid1D "
       << grid_.grida().step() << " " << grid_.gridb().step() << " " << grid_.gridc().step() << endl;


  cout << "Dumping " << grid_.data().size() << " field values " << endl;
  // grid_.dump();
  
}

MFGrid::LocalVector TrapezoidalCartesianMFGrid::valueInTesla( const LocalPoint& p) const
{
//   static TimingReport::Item & timer= (*TimingReport::current())["MagneticFieldProvider::valueInTesla(TrapezoidalCartesianMFGrid)"];
//   TimeMe t(timer,false);

// cout << "TrapezoidalCartesianMFGrid::valueInTesla at local point " << p << endl;

  double xrec, yrec;
  mapping_.rectangle( p.x(), p.y(), xrec, yrec);

//   cout << p.x() << " " << p.y()  
//        << " transformed to grid frame: " << xrec << " " << yrec << endl;

  LinearGridInterpolator3D<GridType::ValueType, GridType::Scalar> interpol( grid_);
  GridType::ValueType value = interpol( xrec, yrec, p.z());
  return LocalVector(value);
}

void TrapezoidalCartesianMFGrid::toGridFrame( const LocalPoint& p, 
					      double& a, double& b, double& c) const
{
  mapping_.rectangle( p.x(), p.y(), a, b);
  c = p.z();
}
 
MFGrid::LocalPoint TrapezoidalCartesianMFGrid::fromGridFrame( double a, double b, double c) const
{
  double xtrap, ytrap;
  mapping_.trapezoid( a, b, xtrap, ytrap);
  return LocalPoint( xtrap, ytrap, c);
}
