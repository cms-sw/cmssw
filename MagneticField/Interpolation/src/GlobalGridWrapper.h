#ifndef GlobalGridWrapper_h
#define GlobalGridWrapper_h

/** \class GlobalGridWrapper
 *
 *  Generic interpolator that is a wrapper of MagneticFieldGrid, i.e.
 *  non-specialized/optimized for each kind of grid.
 *
 *  \author T. Todorov
 */

#include "FWCore/Utilities/interface/Visibility.h"
#include "MagneticField/Interpolation/interface/MFGrid.h"

#include <string>

class MagneticFieldGrid;

class dso_internal GlobalGridWrapper : public MFGrid {
public:
  GlobalGridWrapper(const GloballyPositioned<float>& vol, const std::string& fileName);

  LocalVector valueInTesla(const LocalPoint& p) const override;

  void dump() const override;

  void toGridFrame(const LocalPoint& p, double& a, double& b, double& c) const override;

  LocalPoint fromGridFrame(double a, double b, double c) const override;

  Dimensions dimensions() const override;

  LocalPoint nodePosition(int i, int j, int k) const override;

  LocalVector nodeValue(int i, int j, int k) const override;

private:
  MagneticFieldGrid* theRealOne;
};

#endif
