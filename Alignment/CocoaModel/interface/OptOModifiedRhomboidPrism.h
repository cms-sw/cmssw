//   COCOA class header file
//Id:  OptOModifiedRhomboidPrism.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOMODIFIEDRHOMBOIDPRISM_HH
#define _OPTOMODIFIEDRHOMBOIDPRISM_HH

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOModifiedRhomboidPrism: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOModifiedRhomboidPrism(){ };
  OptOModifiedRhomboidPrism(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOModifiedRhomboidPrism(){ };

  //---------- Fast simulation of deviation of the light ray (reflection, shift, ...)
  virtual void fastDeviatesLightRay( LightRay& lightray );
  //---------- Detailed simulation of the light ray traversing
  virtual void fastTraversesLightRay( LightRay& lightray );
  //---------- Detailed simulation of deviation of the light ray (reflection, shift, ...)
  virtual void detailedDeviatesLightRay( LightRay& lightray );
  //---------- Fast simulation of the light ray traversing
  virtual void detailedTraversesLightRay( LightRay& lightray );

  //--------- Get the up and down plates rotated by an angle 'angle_planes'
  ALIPlane getRotatedPlate(const ALIbool forwardPlate);


#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape();

};

#endif
