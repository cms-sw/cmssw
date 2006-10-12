#ifndef RectangularCylindricalMFGrid_H
#define RectangularCylindricalMFGrid_H

#include "MagneticField/Interpolation/interface/MFGrid3D.h"

class binary_ifstream;

class RectangularCylindricalMFGrid : public MFGrid3D {
public:

  RectangularCylindricalMFGrid( binary_ifstream& istr, 
				const GloballyPositioned<float>& vol );

  virtual LocalVector valueInTesla( const LocalPoint& p) const;

  virtual void dump() const;

  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const;

  virtual LocalPoint fromGridFrame( double a, double b, double c) const;

};

#endif
