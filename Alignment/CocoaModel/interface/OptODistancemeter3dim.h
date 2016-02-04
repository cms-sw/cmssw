//   COCOA class header file
//Id:  OptODistancemeter3dim.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTODISTANCEMETER_HH
#define _OPTODISTANCEMETER_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptODistancemeter3dim: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptODistancemeter3dim(){ };
  OptODistancemeter3dim(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptODistancemeter3dim(){ };

  //---------- defaultBehaviour: make measurement 
  virtual void defaultBehaviour( LightRay& lightray, Measurement& meas );

  //---------- Make measurement 
  virtual void makeMeasurement( LightRay& lightray, Measurement& meas );

#ifdef COCOA_VIS
  virtual void fillVRML();
  virtual void fillIguana();
#endif
  void constructSolidShape();

};

#endif

