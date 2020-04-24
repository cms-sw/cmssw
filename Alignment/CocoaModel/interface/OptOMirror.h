//   COCOA class header file
//Id:  OptOMirror.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOMIRROR_HH
#define _OPTOMIRROR_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOMirror: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOMirror(){ };
  OptOMirror(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOMirror() override{ };

  //---------- Propagate light for measurement meas
  //----- Default behaviour: detailed deviation
  void defaultBehaviour( LightRay& lightray, Measurement& meas ) override;

  void detailedDeviatesLightRay( LightRay& lightray ) override;
 
  void fastDeviatesLightRay( LightRay& lightray ) override;

  void detailedTraversesLightRay( LightRay& lightray ) override;

  void fastTraversesLightRay( LightRay& lightray ) override;

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape() override;

};

#endif

