//   COCOA class header file
//Id:  OptOSource.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOSOURCE_HH
#define _OPTOSOURCE_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOSource: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOSource();
  OptOSource(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOSource() override{ };


#ifdef COCOA_VIS
  virtual void fillVRML();
  virtual void fillIguana();
#endif

  //---------- Propagate light for measurement meas
  void defaultBehaviour( LightRay& lightray, Measurement& meas ) override;
  void constructSolidShape() override;


};

#endif

