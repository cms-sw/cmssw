#ifndef TrapezoidalCylindricalMFGrid_H
#define TrapezoidalCylindricalMFGrid_H

#include "MFGrid3D.h"
#include "Trapezoid2RectangleMappingX.h"
#include "FWCore/Utilities/interface/Visibility.h"


class binary_ifstream;

class dso_internal TrapezoidalCylindricalMFGrid : public MFGrid3D {
public:

  TrapezoidalCylindricalMFGrid( binary_ifstream& istr, 
				const GloballyPositioned<float>& vol);

  virtual LocalVector uncheckedValueInTesla( const LocalPoint& p) const;

  void dump() const;

  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const;

  virtual LocalPoint fromGridFrame( double a, double b, double c) const;

private:

  Trapezoid2RectangleMappingX mapping_;

};

#endif
