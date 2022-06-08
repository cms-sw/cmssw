#ifndef Interpolation_TrapezoidalCartesianMFGrid_h
#define Interpolation_TrapezoidalCartesianMFGrid_h

/** \class TrapezoidalCartesianMFGrid
 *
 *  Grid for a trapezoid in cartesian coordinate.
 *  The grid must have uniform spacing in two coordinates and increasing spacing in the other.
 *  Increasing spacing is supported only for x and y for the time being
 *
 *  \author T. Todorov
 */

#include "MFGrid3D.h"
#include "Trapezoid2RectangleMappingX.h"
#include "FWCore/Utilities/interface/Visibility.h"

namespace magneticfield::interpolation {
  class binary_ifstream;
}

class dso_internal TrapezoidalCartesianMFGrid : public MFGrid3D {
public:
  using binary_ifstream = magneticfield::interpolation::binary_ifstream;

  TrapezoidalCartesianMFGrid(binary_ifstream& istr, const GloballyPositioned<float>& vol);

  LocalVector uncheckedValueInTesla(const LocalPoint& p) const override;

  void dump() const override;

  void toGridFrame(const LocalPoint& p, double& a, double& b, double& c) const override;

  LocalPoint fromGridFrame(double a, double b, double c) const override;

private:
  Trapezoid2RectangleMappingX mapping_;
  bool increasingAlongX;
  bool convertToLocal;
};

#endif
