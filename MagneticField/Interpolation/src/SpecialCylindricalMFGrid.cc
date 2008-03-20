#include "MagneticField/Interpolation/src/SpecialCylindricalMFGrid.h"
#include "MagneticField/Interpolation/src/binary_ifstream.h"
#include "MagneticField/Interpolation/src/LinearGridInterpolator3D.h"

// #include "Utilities/Notification/interface/TimingReport.h"
// #include "Utilities/UI/interface/SimpleConfigurable.h"

#include <iostream>

using namespace std;

SpecialCylindricalMFGrid::SpecialCylindricalMFGrid( binary_ifstream& inFile, 
						    const GloballyPositioned<float>& vol)
  : MFGrid3D(vol)
{
  int n1, n2, n3;
  inFile >> n1 >> n2 >> n3;
#ifdef DEBUG_GRID
  cout << "n1 " << n1 << " n2 " << n2 << " n3 " << n3 << endl;
#endif
  double xref, yref, zref;
  inFile >> xref >> yref >> zref;
  double stepx, stepy, stepz;
  inFile >> stepx    >> stepy    >> stepz;

  double RParAsFunOfPhi[4];  // R = f(phi) or const. (0,2: const. par. ; 1,3: const./sin(phi));
  inFile >> RParAsFunOfPhi[0] >> RParAsFunOfPhi[1] >> RParAsFunOfPhi[2] >> RParAsFunOfPhi[3];

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

#ifdef DEBUG_GRID
  LocalPoint lrefp = frame().toLocal( grefp);
  cout << "Grid reference point in grid system: " << xref << "," << yref << "," << zref << endl;
  cout << "Grid reference point in global x,y,z: " << grefp << endl;
  cout << "Grid reference point in local x,y,z: " << lrefp << endl;
  cout << "steps " << stepx << "," <<  stepy << "," << stepz << endl;
  cout << "RParAsFunOfPhi[0...4] = ";
  for (int i=0; i<4; ++i) cout << RParAsFunOfPhi[i] << " "; cout << endl;
#endif

  Grid1D<double> gridX( 0, n1-1, n1); // offset and step size not constant
  Grid1D<double> gridY( yref, yref + stepy*(n2-1), n2);
  Grid1D<double> gridZ( grefp.z(), grefp.z() + stepz*(n3-1), n3);

  grid_ = GridType( gridX, gridY, gridZ, fieldValues);

  stepConstTerm_ = (RParAsFunOfPhi[0] - RParAsFunOfPhi[2]) / (n1-1);
  stepPhiTerm_   = (RParAsFunOfPhi[1] - RParAsFunOfPhi[3]) / (n1-1);
  startConstTerm_ = RParAsFunOfPhi[2];
  startPhiTerm_   = RParAsFunOfPhi[3];

  // Activate/deactivate timers
//   static SimpleConfigurable<bool> timerOn(false,"MFGrid:timing");
//   (*TimingReport::current()).switchOn("MagneticFieldProvider::valueInTesla(SpecialCylindricalMFGrid)",timerOn);
}
 

MFGrid::LocalVector SpecialCylindricalMFGrid::uncheckedValueInTesla( const LocalPoint& p) const
{
//   static TimingReport::Item & timer= (*TimingReport::current())["MagneticFieldProvider::valueInTesla(SpecialCylindricalMFGrid)"];
//   TimeMe t(timer,false);

  LinearGridInterpolator3D<GridType::ValueType, GridType::Scalar> interpol( grid_);
  double a, b, c;
  toGridFrame( p, a, b, c);
  GlobalVector gv( interpol( a, b, c)); // grid in global frame
  return frame().toLocal(gv);           // must return a local vector
}

void SpecialCylindricalMFGrid::dump() const {}


MFGrid::LocalPoint SpecialCylindricalMFGrid::fromGridFrame( double a, double b, double c) const
{
  double sinPhi = sin(b);
  double R = a*stepSize(sinPhi) + startingPoint(sinPhi);
  // FIXME: "OLD" convention of phi.
  //  GlobalPoint gp( GlobalPoint::Cylindrical(R, Geom::pi() - b, c));
  GlobalPoint gp( GlobalPoint::Cylindrical(R, b, c));
  return frame().toLocal(gp);
}

void SpecialCylindricalMFGrid::toGridFrame( const LocalPoint& p, 
					    double& a, double& b, double& c) const
{
  GlobalPoint gp = frame().toGlobal(p);
  double sinPhi = sin(gp.phi());
  a = (gp.perp()-startingPoint(sinPhi))/stepSize(sinPhi);
  // FIXME: "OLD" convention of phi.
  // b = Geom::pi() - gp.phi();
  b = gp.phi();
  c = gp.z();

#ifdef DEBUG_GRID
  cout << "toGridFrame: sinPhi " << sinPhi << " LocalPoint " << p 
       << " GlobalPoint " << gp << endl 
       << " a " << a << " b " << b << " c " << c << endl;
#endif
}

