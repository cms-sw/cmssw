#ifndef Interpolation_TrapezoidalCartesianMFGrid_h
#define Interpolation_TrapezoidalCartesianMFGrid_h

/** \class TrapezoidalCartesianMFGrid
 *
 *  Grid for a trapezoid in cartesian coordinate.
 *  The grid must have uniform spacing in two coordinates and increasing spacing in the other.
 *  Increasing spacing is supported only for x and y for the time being
 *
 *  $Date: $
 *  $Revision: $
 *  \author T. Todorov
 */

#include "MagneticField/Interpolation/interface/MFGrid3D.h"
#include "MagneticField/Interpolation/src/Trapezoid2RectangleMappingX.h"

class binary_ifstream;

class TrapezoidalCartesianMFGrid : public MFGrid3D {
public:

  TrapezoidalCartesianMFGrid( binary_ifstream& istr, 
			      const GloballyPositioned<float>& vol);

  virtual LocalVector uncheckedValueInTesla( const LocalPoint& p) const;

  void dump() const;

  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const;

  virtual LocalPoint fromGridFrame( double a, double b, double c) const;

private:

  Trapezoid2RectangleMappingX mapping_;
  bool increasingAlongX;
  bool convertToLocal;
  
};

#endif
