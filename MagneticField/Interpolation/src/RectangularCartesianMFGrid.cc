#include "RectangularCartesianMFGrid.h"
#include "MagneticField/Interpolation/interface/binary_ifstream.h"
#include "LinearGridInterpolator3D.h"

#include <iostream>

using namespace std;

RectangularCartesianMFGrid::RectangularCartesianMFGrid(binary_ifstream& inFile, const GloballyPositioned<float>& vol)
    : MFGrid3D(vol) {
  // The parameters read from the data files are given in global coordinates.
  // In version 85l, local frame has the same orientation of global frame for the reference
  // volume, i.e. the r.f. transformation is only a translation.
  // There is therefore no need to convert the field values to local coordinates.
  // Check this assumption:
  GlobalVector localXDir(frame().toGlobal(LocalVector(1, 0, 0)));
  GlobalVector localYDir(frame().toGlobal(LocalVector(0, 1, 0)));

  if (localXDir.dot(GlobalVector(1, 0, 0)) > 0.999999 && localYDir.dot(GlobalVector(0, 1, 0)) > 0.999999) {
    // "null" rotation - requires no conversion...
  } else {
    cout << "ERROR: RectangularCartesianMFGrid: unexpected orientation: x: " << localXDir << " y: " << localYDir
         << endl;
  }

  int n1, n2, n3;
  inFile >> n1 >> n2 >> n3;
  double xref, yref, zref;
  inFile >> xref >> yref >> zref;
  double stepx, stepy, stepz;
  inFile >> stepx >> stepy >> stepz;

  vector<BVector> fieldValues;
  float Bx, By, Bz;
  int nLines = n1 * n2 * n3;
  fieldValues.reserve(nLines);
  for (int iLine = 0; iLine < nLines; ++iLine) {
    inFile >> Bx >> By >> Bz;
    fieldValues.push_back(BVector(Bx, By, Bz));
  }
  // check completeness
  string lastEntry;
  inFile >> lastEntry;
  if (lastEntry != "complete") {
    cout << "ERROR during file reading: file is not complete" << endl;
  }

  GlobalPoint grefp(xref, yref, zref);
  LocalPoint lrefp = frame().toLocal(grefp);

  Grid1D gridX(lrefp.x(), lrefp.x() + stepx * (n1 - 1), n1);
  Grid1D gridY(lrefp.y(), lrefp.y() + stepy * (n2 - 1), n2);
  Grid1D gridZ(lrefp.z(), lrefp.z() + stepz * (n3 - 1), n3);
  grid_ = GridType(gridX, gridY, gridZ, fieldValues);

  // Activate/deactivate timers
  //   static SimpleConfigurable<bool> timerOn(false,"MFGrid:timing");
  //   (*TimingReport::current()).switchOn("MagneticFieldProvider::valueInTesla(RectangularCartesianMFGrid)",timerOn);
}

void RectangularCartesianMFGrid::dump() const {
  cout << endl << "Dump of RectangularCartesianMFGrid" << endl;
  //   cout << "Number of points from file   "
  //        << n1 << " " << n2 << " " << n3 << endl;
  cout << "Number of points from Grid1D " << grid_.grida().nodes() << " " << grid_.gridb().nodes() << " "
       << grid_.gridc().nodes() << endl;

  //   cout << "Reference Point from file   "
  //        << xref << " " << yref << " " << zref << endl;
  cout << "Reference Point from Grid1D " << grid_.grida().lower() << " " << grid_.gridb().lower() << " "
       << grid_.gridc().lower() << endl;

  //   cout << "Basic Distance from file   "
  //        <<  stepx << " " << stepy << " " << stepz << endl;
  cout << "Basic Distance from Grid1D " << grid_.grida().step() << " " << grid_.gridb().step() << " "
       << grid_.gridc().step() << endl;

  cout << "Dumping " << grid_.data().size() << " field values " << endl;
  // grid_.dump();
}

MFGrid::LocalVector RectangularCartesianMFGrid::uncheckedValueInTesla(const LocalPoint& p) const {
  //   static TimingReport::Item & timer= (*TimingReport::current())["MagneticFieldProvider::valueInTesla(RectangularCartesianMFGrid)"];
  //   TimeMe t(timer,false);

  LinearGridInterpolator3D interpol(grid_);
  GridType::ReturnType value = interpol.interpolate(p.x(), p.y(), p.z());
  return LocalVector(value);
}

void RectangularCartesianMFGrid::toGridFrame(const LocalPoint& p, double& a, double& b, double& c) const {
  a = p.x();
  b = p.y();
  c = p.z();
}

MFGrid::LocalPoint RectangularCartesianMFGrid::fromGridFrame(double a, double b, double c) const {
  return LocalPoint(a, b, c);
}
