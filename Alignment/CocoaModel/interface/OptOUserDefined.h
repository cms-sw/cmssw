//   COCOA class header file
//Id:  OptOUserDefined.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
//
//   History: v1.0
//   Pedro Arce

#ifndef _OPTOUSERDEFINED_HH
#define _OPTOUSERDEFINED_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOUserDefined : public OpticalObject {
public:
  //---------- Constructors / Destructor
  OptOUserDefined(){};
  OptOUserDefined(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data)
      : OpticalObject(parent, type, name, copy_data){};
  ~OptOUserDefined() override{};

#ifdef COCOA_VIS
  virtual void fillVRML() override;
  virtual void fillIguana() override;
#endif
  //---------- userDefinedBehaviour
  void userDefinedBehaviour(LightRay& lightray, Measurement& meas, const ALIstring& behav) override;
};

#endif
