//   COCOA class header file
//Id:  OptOLaser.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOLASER_HH
#define _OPTOLASER_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOLaser: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOLaser(){ };
  OptOLaser(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOLaser(){ };

  //---------- Default behaviour: create a LightRay object
  virtual void defaultBehaviour( LightRay& lightray, Measurement& meas );

#ifdef COCOA_VIS
  virtual void fillVRML();
  virtual void fillIguana();
#endif
  void constructSolidShape();

};

#endif
