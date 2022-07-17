#include "SpecialCylindricalMFGrid.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "LinearGridInterpolator3D.h"

#include <iostream>

using namespace std;

SpecialCylindricalMFGrid::SpecialCylindricalMFGrid(binary_ifstream& inFile,
                                                   const GloballyPositioned<float>& vol,
                                                   int gridType)
    : MFGrid3D(vol) {
  if (gridType == 5) {
    sector1 = false;
  } else if (gridType == 6) {
    sector1 = true;
  } else {
    cout << "ERROR wrong SpecialCylindricalMFGrid type " << gridType << endl;
    sector1 = false;
  }

  int n1, n2, n3;
  inFile >> n1 >> n2 >> n3;
#ifdef DEBUG_GRID
  cout << "n1 " << n1 << " n2 " << n2 << " n3 " << n3 << endl;
#endif
  double xref, yref, zref;
  inFile >> xref >> yref >> zref;
  double stepx, stepy, stepz;
  inFile >> stepx >> stepy >> stepz;

  double RParAsFunOfPhi[4];  // R = f(phi) or const. (0,2: const. par. ; 1,3: const./sin(phi));
  inFile >> RParAsFunOfPhi[0] >> RParAsFunOfPhi[1] >> RParAsFunOfPhi[2] >> RParAsFunOfPhi[3];

  vector<BVector> fieldValues;
  float Bx, By, Bz;
  int nLines = n1 * n2 * n3;
  fieldValues.reserve(nLines);
  for (int iLine = 0; iLine < nLines; ++iLine) {
    inFile >> Bx >> By >> Bz;
    // This would be fine only if local r.f. has the axes oriented as the global r.f.
    // For this volume we know that the local and global r.f. have different axis
    // orientation, so we do not try to be clever.
    //    fieldValues.push_back(BVector(Bx,By,Bz));

    // Preserve double precision!
    Vector3DBase<double, LocalTag> lB = frame().toLocal(Vector3DBase<double, GlobalTag>(Bx, By, Bz));
    fieldValues.push_back(BVector(lB.x(), lB.y(), lB.z()));
  }
  // check completeness
  string lastEntry;
  inFile >> lastEntry;
  if (lastEntry != "complete") {
    cout << "ERROR during file reading: file is not complete" << endl;
  }

  GlobalPoint grefp(GlobalPoint::Cylindrical(xref, yref, zref));

#ifdef DEBUG_GRID
  LocalPoint lrefp = frame().toLocal(grefp);
  cout << "Grid reference point in grid system: " << xref << "," << yref << "," << zref << endl;
  cout << "Grid reference point in global x,y,z: " << grefp << endl;
  cout << "Grid reference point in local x,y,z: " << lrefp << endl;
  cout << "steps " << stepx << "," << stepy << "," << stepz << endl;
  cout << "RParAsFunOfPhi[0...4] = ";
  for (int i = 0; i < 4; ++i)
    cout << RParAsFunOfPhi[i] << " ";
  cout << endl;
#endif

  Grid1D gridX(0, n1 - 1, n1);  // offset and step size not constant
  Grid1D gridY(yref, yref + stepy * (n2 - 1), n2);
  Grid1D gridZ(grefp.z(), grefp.z() + stepz * (n3 - 1), n3);

  grid_ = GridType(gridX, gridY, gridZ, fieldValues);

  stepConstTerm_ = (RParAsFunOfPhi[0] - RParAsFunOfPhi[2]) / (n1 - 1);
  stepPhiTerm_ = (RParAsFunOfPhi[1] - RParAsFunOfPhi[3]) / (n1 - 1);
  startConstTerm_ = RParAsFunOfPhi[2];
  startPhiTerm_ = RParAsFunOfPhi[3];

  // Activate/deactivate timers
  //   static SimpleConfigurable<bool> timerOn(false,"MFGrid:timing");
  //   (*TimingReport::current()).switchOn("MagneticFieldProvider::valueInTesla(SpecialCylindricalMFGrid)",timerOn);
}

MFGrid::LocalVector SpecialCylindricalMFGrid::uncheckedValueInTesla(const LocalPoint& p) const {
  //   static TimingReport::Item & timer= (*TimingReport::current())["MagneticFieldProvider::valueInTesla(SpecialCylindricalMFGrid)"];
  //   TimeMe t(timer,false);

  LinearGridInterpolator3D interpol(grid_);
  double a, b, c;
  toGridFrame(p, a, b, c);
  // the following holds if B values was not converted to local coords -- see ctor
  //   GlobalVector gv( interpol.interpolate( a, b, c)); // grid in global frame
  //   return frame().toLocal(gv);           // must return a local vector
  return LocalVector(interpol.interpolate(a, b, c));
}

void SpecialCylindricalMFGrid::dump() const {}

MFGrid::LocalPoint SpecialCylindricalMFGrid::fromGridFrame(double a, double b, double c) const {
  double sinPhi;  // sin or cos depending on wether we are at phi=0 or phi=pi/2
  if (sector1) {
    sinPhi = cos(b);
  } else {
    sinPhi = sin(b);
  }

  double R = a * stepSize(sinPhi) + startingPoint(sinPhi);
  // "OLD" convention of phi.
  //  GlobalPoint gp( GlobalPoint::Cylindrical(R, Geom::pi() - b, c));
  GlobalPoint gp(GlobalPoint::Cylindrical(R, b, c));
  return frame().toLocal(gp);
}

void SpecialCylindricalMFGrid::toGridFrame(const LocalPoint& p, double& a, double& b, double& c) const {
  GlobalPoint gp = frame().toGlobal(p);
  double sinPhi;  // sin or cos depending on wether we are at phi=0 or phi=pi/2
  if (sector1) {
    sinPhi = cos(gp.phi());
  } else {
    sinPhi = sin(gp.phi());
  }
  a = (gp.perp() - startingPoint(sinPhi)) / stepSize(sinPhi);
  // FIXME: "OLD" convention of phi.
  // b = Geom::pi() - gp.phi();
  b = gp.phi();
  c = gp.z();

#ifdef DEBUG_GRID
  if (sector1) {
    cout << "toGridFrame: sinPhi ";
  } else {
    cout << "toGridFrame: cosPhi ";
  }
  cout << sinPhi << " LocalPoint " << p << " GlobalPoint " << gp << endl
       << " a " << a << " b " << b << " c " << c << endl;

#endif
}
