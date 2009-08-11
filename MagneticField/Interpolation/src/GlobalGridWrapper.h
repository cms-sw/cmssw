#ifndef GlobalGridWrapper_h
#define GlobalGridWrapper_h

/** \class GlobalGridWrapper
 *
 *  Generic interpolator that is a wrapper of MagneticFieldGrid, i.e.
 *  non-specialized/optimized for each kind of grid.
 *
 *  $Date: 2008/04/09 19:40:33 $
 *  $Revision: 1.2 $
 *  \author T. Todorov
 */


#include "MagneticField/Interpolation/interface/MFGrid.h"

#include <string>

class binary_ifstream;
class MagneticFieldGrid;

class GlobalGridWrapper : public MFGrid {
public:

  GlobalGridWrapper(  const GloballyPositioned<float>& vol,
		      const std::string& fileName);

  virtual LocalVector valueInTesla( const LocalPoint& p) const;

  void dump() const;

  virtual void toGridFrame( const LocalPoint& p, double& a, double& b, double& c) const;

  virtual LocalPoint fromGridFrame( double a, double b, double c) const;

  virtual Dimensions dimensions() const;

  virtual LocalPoint  nodePosition( int i, int j, int k) const;

  virtual LocalVector nodeValue( int i, int j, int k) const;

private:

  MagneticFieldGrid* theRealOne;

};

#endif
