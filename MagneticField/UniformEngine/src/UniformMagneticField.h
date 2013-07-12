#ifndef MagneticField_UniformMagneticField_h
#define MagneticField_UniformMagneticField_h

/** \class UniformMagneticField
 *
 *  A MagneticField engine that returns a constant programmable field value.
 *
 *  $Date: 2008/03/29 11:54:58 $
 *  $Revision: 1.2 $
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
