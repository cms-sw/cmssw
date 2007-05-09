#ifndef SpecialCylindricalMFGrid_H
#define SpecialCylindricalMFGrid_H

#include "MagneticField/Interpolation/interface/MFGrid3D.h"

class binary_ifstream;

class SpecialCylindricalMFGrid : public MFGrid3D {
public:

  SpecialCylindricalMFGrid( binary_ifstream& istr, 
			    const GloballyPositioned<float>& vol );

  virtual LocalVector valueInTesla( const LocalPoint& p) const;

  virtual void dump() const;

  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const;

  virtual LocalPoint fromGridFrame( double a, double b, double c) const;

private:

  //double RParAsFunOfPhi[4];     // R = f(phi) or const. (0,2: const. par. ; 1,3: const./sin(phi))

  double stepConstTerm_;
  double stepPhiTerm_;
  double startConstTerm_;
  double startPhiTerm_;

  double stepSize( double sinPhi) const {return stepConstTerm_ + stepPhiTerm_ /sinPhi;}
  double startingPoint( double sinPhi) const {return startConstTerm_ + startPhiTerm_/sinPhi;}

};

#endif
