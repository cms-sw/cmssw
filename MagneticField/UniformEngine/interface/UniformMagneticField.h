#ifndef MagneticField_UniformMagneticField_h
#define MagneticField_UniformMagneticField_h

/** \class UniformMagneticField
 *
 *  A MagneticField engine that returns a constant programmable field value.
 *
 *  \author N. Amapane - CERN
 */


#include "MagneticField/Engine/interface/MagneticField.h"

class UniformMagneticField final : public MagneticField {
 public:

  ///Construct passing the Z field component in Tesla
  UniformMagneticField(float value) : theField(0.f,0.f,value) {}

  UniformMagneticField(GlobalVector value) :  theField(value) {}

  void set(GlobalVector value) { theField =value;}
  void set(float value) { set(GlobalVector(0.f,0.f,value)); }


  virtual ~UniformMagneticField() {}

  GlobalVector inTesla (const GlobalPoint&) const override {return theField;}

  GlobalVector inTeslaUnchecked (const GlobalPoint& gp) const override {return theField;}

  bool isDefined(const GlobalPoint& gp) const override {return true;}

 private:
  GlobalVector theField;
};

#endif
