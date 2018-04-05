//   COCOA class header file
//Id:  OptOScreen.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOSCREEN_HH
#define _OPTOSCREEN_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOScreen: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOScreen(){ };
  OptOScreen(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOScreen() override{ };

  //---------- defaultBehaviour: do nothing
  void defaultBehaviour( LightRay& lightray, Measurement& meas ) override;
#ifdef COCOA_VIS
  virtual void fillVRML();
  virtual void fillIguana();
#endif
  void constructSolidShape() override;

};

#endif

