#ifndef GlobalGridWrapper_h
#define GlobalGridWrapper_h

/** \class GlobalGridWrapper
 *
 *  Generic interpolator that is a wrapper of MagneticFieldGrid, i.e.
 *  non-specialized/optimized for each kind of grid.
 *
 *  $Date: 2011/04/16 12:47:37 $
 *  $Revision: 1.4 $
 *  \author T. Todorov
 */


#include "FWCore/Utilities/interface/Visibility.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"

#include <string>

class binary_ifstream;
class MagneticFieldGrid;

class dso_internal GlobalGridWrapper : public MFGrid {
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
