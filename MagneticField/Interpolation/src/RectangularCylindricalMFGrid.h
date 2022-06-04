#ifndef RectangularCylindricalMFGrid_H
#define RectangularCylindricalMFGrid_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "MFGrid3D.h"

namespace magneticfield::interpolation {
  class binary_ifstream;
}

class dso_internal RectangularCylindricalMFGrid : public MFGrid3D {
public:
  using binary_ifstream = magneticfield::interpolation::binary_ifstream;

  RectangularCylindricalMFGrid(binary_ifstream& istr, const GloballyPositioned<float>& vol);

  LocalVector uncheckedValueInTesla(const LocalPoint& p) const override;

  void dump() const override;

  void toGridFrame(const LocalPoint& p, double& a, double& b, double& c) const override;

  LocalPoint fromGridFrame(double a, double b, double c) const override;
};

#endif
