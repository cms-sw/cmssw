#ifndef MagneticField_MagneticField_h
#define MagneticField_MagneticField_h

/** \class MagneticField
 *
 *  Base class for the different implementation of magnetic field engines.
 *
 *  $Date: 2009/05/23 22:57:42 $
 *  $Revision: 1.8 $
 *  \author N. Amapane - CERN
 */

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class MagneticField
{
 public:
  MagneticField();
  virtual ~MagneticField();

  /// Derived classes can implement cloning without ownership of the 
  /// underlying engines.
  virtual MagneticField* clone() const {
    return 0;
  }
  

  /// Field value ad specified global point, in Tesla
  virtual GlobalVector inTesla (const GlobalPoint& gp) const = 0;

  /// Field value ad specified global point, in KGauss
  GlobalVector inKGauss(const GlobalPoint& gp) const  {
    return inTesla(gp) * 10.F;
  }

  /// Field value ad specified global point, in 1/Gev
  GlobalVector inInverseGeV(const GlobalPoint& gp) const {
    return inTesla(gp) * 2.99792458e-3F;
  }

  /// True if the point is within the region where the concrete field
  // engine is defined.
  virtual bool isDefined(const GlobalPoint& gp) const {
    return true;
  }
  
  /// Optional implementation that derived classes can implement to provide faster query
  /// by skipping the check to isDefined.
  virtual GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const {
    return inTesla(gp);  // default dummy implementation
  }
  
  /// The nominal field value for this map in kGauss
  // This generic implementation can be replaced by concrete engines
  virtual int nominalValue() const;
  
};

#endif
