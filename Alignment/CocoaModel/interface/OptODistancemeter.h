//   COCOA class header file
//Id:  OptODistancemeter.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
//
//   History: v1.0
//   Pedro Arce

#ifndef _OPTODISTANCEMETER1DIM_HH
#define _OPTODISTANCEMETER1DIM_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptODistancemeter : public OpticalObject {
public:
  //---------- Constructors / Destructor
  OptODistancemeter(){};
  OptODistancemeter(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data)
      : OpticalObject(parent, type, name, copy_data){};
  ~OptODistancemeter() override{};

  //---------- defaultBehaviour: make measurement
  void defaultBehaviour(LightRay& lightray, Measurement& meas) override;

  //---------- Make measurement
  void makeMeasurement(LightRay& lightray, Measurement& meas) override;

#ifdef COCOA_VIS
  void fillVRML();
  virtual void fillIguana();
#endif
  void constructSolidShape() override;
};

#endif
