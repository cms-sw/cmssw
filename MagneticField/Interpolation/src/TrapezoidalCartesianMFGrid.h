#ifndef Interpolation_TrapezoidalCartesianMFGrid_h
#define Interpolation_TrapezoidalCartesianMFGrid_h

/** \class TrapezoidalCartesianMFGrid
 *
 *  Grid for a trapezoid in cartesian coordinate.
 *  The grid must have uniform spacing in two coordinates and increasing spacing in the other.
 *  Increasing spacing is supported only for x and y for the time being
 *
 *  $Date: 2011/04/16 12:47:37 $
 *  $Revision: 1.3 $
 *  \author T. Todorov
 */

#include "MFGrid3D.h"
#include "Trapezoid2RectangleMappingX.h"
#include "FWCore/Utilities/interface/Visibility.h"

class binary_ifstream;

class dso_internal TrapezoidalCartesianMFGrid : public MFGrid3D {
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
