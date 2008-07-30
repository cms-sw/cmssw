#ifndef MagneticField_UniformMagneticField_h
#define MagneticField_UniformMagneticField_h

/** \class UniformMagneticField
 *
 *  A MagneticField engine that returns a constant programmable field value.
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */


#include "MagneticField/Engine/interface/MagneticField.h"

class UniformMagneticField : public MagneticField {
 public:

  ///Construct passing the Z field component in Tesla
  UniformMagneticField(double value);

  virtual ~UniformMagneticField() {}

  GlobalVector inTesla (const GlobalPoint& gp) const;

 private:
  GlobalVector theField;
};

#endif
