#include "MagneticField/Interpolation/interface/RectangularCartesianMFGrid.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "MagneticField/Interpolation/interface/LinearGridInterpolator3D.h"

// #include "Utilities/Notification/interface/TimingReport.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"

#include <iostream>

using namespace std;

RectangularCartesianMFGrid::RectangularCartesianMFGrid( binary_ifstream& inFile,
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

  GlobalPoint grefp( xref, yref, zref);
  LocalPoint lrefp = frame().toLocal( grefp);

  Grid1D<double> gridX( lrefp.x(), lrefp.x() + stepx*(n1-1), n1);
  Grid1D<double> gridY( lrefp.y(), lrefp.y() + stepy*(n2-1), n2);
  Grid1D<double> gridZ( lrefp.z(), lrefp.z() + stepz*(n3-1), n3);
  grid_ = GridType( gridX, gridY, gridZ, fieldValues);
  
  // Activate/deactivate timers
//   static SimpleConfigurable<bool> timerOn(false,"MFGrid:timing");
//   (*TimingReport::current()).switchOn("MagneticFieldProvider::valueInTesla(RectangularCartesianMFGrid)",timerOn);

}

void RectangularCartesianMFGrid::dump() const
{
  cout << endl << "Dump of RectangularCartesianMFGrid" << endl;
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

MFGrid::LocalVector RectangularCartesianMFGrid::valueInTesla( const LocalPoint& p) const
{
//   static TimingReport::Item & timer= (*TimingReport::current())["MagneticFieldProvider::valueInTesla(RectangularCartesianMFGrid)"];
//   TimeMe t(timer,false);

  LinearGridInterpolator3D<GridType::ValueType, GridType::Scalar> interpol( grid_);
  GridType::ValueType value = interpol( p.x(), p.y(), p.z());
  return LocalVector(value);
}

void RectangularCartesianMFGrid::toGridFrame( const LocalPoint& p, 
					      double& a, double& b, double& c) const
{
  a = p.x();
  b = p.y();
  c = p.z();
}
 
MFGrid::LocalPoint RectangularCartesianMFGrid::fromGridFrame( double a, double b, double c) const
{
  return LocalPoint( a, b, c);
}
