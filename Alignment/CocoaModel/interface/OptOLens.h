//   COCOA class header file
//Id:  OptOLens.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
//
//   History: v1.0
//   Pedro Arce

#ifndef _OPTOLENS_HH
#define _OPTOLENS_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOLens : public OpticalObject {
public:
  //---------- Constructors / Destructor
  OptOLens();
  OptOLens(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data)
      : OpticalObject(parent, type, name, copy_data){};
  ~OptOLens() override{};

  //---------- Propagate light for measurement meas
  void participateInMeasurement(LightRay& lightray, Measurement& meas, const ALIstring& behav) override;

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape() override;
};

#endif
