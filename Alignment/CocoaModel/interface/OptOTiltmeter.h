//   COCOA class header file
//Id:  OptOTiltmeter.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOTILTMETER_HH
#define _OPTOTILTMETER_HH

#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "OpticalAlignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOTiltmeter: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOTiltmeter(){ };
  OptOTiltmeter(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOTiltmeter(){ };

  //---------- defaultBehaviour: make measurement 
  virtual void defaultBehaviour( LightRay& lightray, Measurement& meas );
  //---------- Make measurement 
  virtual void makeMeasurement( LightRay& lightray, Measurement& meas );
#ifdef COCOA_VIS
  virtual void fillVRML();
  virtual void fillIguana();
#endif

};

#endif

