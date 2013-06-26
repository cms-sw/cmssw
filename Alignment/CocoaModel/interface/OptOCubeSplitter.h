//   COCOA class header file
//Id:  OptOPlateSplitter.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOCUBESPLITTER_HH
#define _OPTOCUBESPLITTER_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOCubeSplitter: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOCubeSplitter(){ };
  OptOCubeSplitter(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOCubeSplitter(){ };

  //---------- Fast simulation of deviation of the light ray (reflection, shift, ...)
  virtual void fastDeviatesLightRay( LightRay& lightray );
  //---------- Detailed simulation of the light ray traversing
  virtual void fastTraversesLightRay( LightRay& lightray );
  //---------- Detailed simulation of deviation of the light ray (reflection, shift, ...)
  virtual void detailedDeviatesLightRay( LightRay& lightray );
  //---------- Fast simulation of the light ray traversing
  virtual void detailedTraversesLightRay( LightRay& lightray );

  ALIPlane getMiddlePlate();
  ALIPlane getUpperPlate();

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape();

};

#endif

