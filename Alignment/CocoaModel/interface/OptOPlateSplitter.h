//   COCOA class header file
//Id:  OptOPlateSplitter.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOPLATESPLITTER_HH
#define _OPTOPLATESPLITTER_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOPlateSplitter: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOPlateSplitter(){ };
  OptOPlateSplitter(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOPlateSplitter() override{ };

  //---------- Detailed simulation of deviation of the light ray (reflection, shift, ...)
  void detailedDeviatesLightRay( LightRay& lightray ) override;
  //---------- Detailed simulation of the light ray traversing
  void detailedTraversesLightRay( LightRay& lightray ) override;
  //---------- Fast simulation of deviation of the light ray (reflection, shift, ...)
  void fastDeviatesLightRay( LightRay& lightray ) override;
  //---------- Fast simulation of the light ray traversing
  void fastTraversesLightRay( LightRay& lightray ) override;

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape() override;

};

#endif

