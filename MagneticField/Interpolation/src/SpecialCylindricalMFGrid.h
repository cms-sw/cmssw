#ifndef Interpolation_SpecialCylindricalMFGrid_h
#define Interpolation_SpecialCylindricalMFGrid_h

/** \class SpecialCylindricalMFGrid
 *
 *  Interpolator for cylindrical grids type 5 or 6 (r,phi,z) 1/sin(phi) or 1/cos(phi)
 *
 *  $Date: 2011/04/16 12:47:37 $
 *  $Revision: 1.3 $
 *  \author T. Todorov - updated 08 N. Amapane
 */


#include "FWCore/Utilities/interface/Visibility.h"
#include "MFGrid3D.h"

class binary_ifstream;

class dso_internal SpecialCylindricalMFGrid : public MFGrid3D {
public:

  /// Constructor. 
  /// gridType = 5 => 1/sin(phi); i.e. master sector is #4 
  /// gridType = 6 => 1/cos(phi); i.e. master sector is #1 
  SpecialCylindricalMFGrid( binary_ifstream& istr, 
			    const GloballyPositioned<float>& vol,
			    int gridType);

  virtual LocalVector uncheckedValueInTesla( const LocalPoint& p) const;

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
  bool sector1;
};

#endif
