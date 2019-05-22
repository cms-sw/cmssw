//   COCOA class header file
//Id:  OptOPinhole.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
//
//   History: v1.0
//   Pedro Arce

#ifndef _OPTOPINHOLE_HH
#define _OPTOPINHOLE_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOPinhole : public OpticalObject {
public:
  //---------- Constructors / Destructor
  OptOPinhole(){};
  OptOPinhole(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data)
      : OpticalObject(parent, type, name, copy_data){};
  ~OptOPinhole() override{};

  //---------- Default behaviour
  void defaultBehaviour(LightRay& lightray, Measurement& meas) override;

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape() override;
};

#endif
