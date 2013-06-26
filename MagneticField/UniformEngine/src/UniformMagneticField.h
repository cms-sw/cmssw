#ifndef MagneticField_UniformMagneticField_h
#define MagneticField_UniformMagneticField_h

/** \class UniformMagneticField
 *
 *  A MagneticField engine that returns a constant programmable field value.
 *
 *  $Date: 2008/05/06 12:40:04 $
 *  $Revision: 1.3 $
 *  \author N. Amapane - CERN
 */


#include "MagneticField/Engine/interface/MagneticField.h"

class UniformMagneticField : public MagneticField {
 public:

  ///Construct passing the Z field component in Tesla
  UniformMagneticField(double value);

  virtual ~UniformMagneticField() {}

  GlobalVector inTesla (const GlobalPoint& gp) const;

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const;

  bool isDefined(const GlobalPoint& gp) const {return true;}

 private:
  GlobalVector theField;
};

#endif
