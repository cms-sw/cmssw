#ifndef TrapezoidalCylindricalMFGrid_H
#define TrapezoidalCylindricalMFGrid_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "MFGrid3D.h"
#include "Trapezoid2RectangleMappingX.h"

class binary_ifstream;

class dso_internal TrapezoidalCylindricalMFGrid : public MFGrid3D {
public:
  TrapezoidalCylindricalMFGrid(binary_ifstream &istr,
                               const GloballyPositioned<float> &vol);

  LocalVector uncheckedValueInTesla(const LocalPoint &p) const override;

  void dump() const override;

  void toGridFrame(const LocalPoint &p, double &a, double &b,
                   double &c) const override;

  LocalPoint fromGridFrame(double a, double b, double c) const override;

private:
  Trapezoid2RectangleMappingX mapping_;
};

#endif
