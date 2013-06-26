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
  ~OptOMirror(){ };

  //---------- Propagate light for measurement meas
  //----- Default behaviour: detailed deviation
  void defaultBehaviour( LightRay& lightray, Measurement& meas );

  virtual void detailedDeviatesLightRay( LightRay& lightray );
 
  virtual void fastDeviatesLightRay( LightRay& lightray );

  virtual void detailedTraversesLightRay( LightRay& lightray );

  virtual void fastTraversesLightRay( LightRay& lightray );

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape();

};

#endif

