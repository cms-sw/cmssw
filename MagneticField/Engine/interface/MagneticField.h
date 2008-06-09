#ifndef MagneticField_MagneticField_h
#define MagneticField_MagneticField_h

/** \class MagneticField
 *
 *  Base class for the different implementation of magnetic field engines.
 *
 *  $Date: 2006/05/31 13:42:30 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class MagneticField
{
 public:
  MagneticField() {;}
  virtual ~MagneticField() {;}

  /// Field value ad specified global point, in Tesla
  virtual GlobalVector inTesla (const GlobalPoint& gp) const = 0;

  /// Field value ad specified global point, in KGauss
  virtual GlobalVector inKGauss(const GlobalPoint& gp) const {
    return inTesla(gp) * 10.;
  }

  /// Field value ad specified global point, in 1/Gev
  virtual GlobalVector inInverseGeV(const GlobalPoint& gp) const {
    return inTesla(gp) * 2.99792458e-3;
  }
};

#endif
