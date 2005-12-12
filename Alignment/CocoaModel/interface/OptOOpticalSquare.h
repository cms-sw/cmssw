//   COCOA class header file
//Id:  OptOOpticalSquare.h
//CAT: Model
//
//   Base class to describe Optical Objects of type sensor 2D
// 
//   History: v1.0 
//   Pedro Arce

#ifndef _OPTOPSEUDOPENTAPRISM_HH
#define _OPTOPSEUDOPENTAPRISM_HH

#include "OpticalAlignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "OpticalAlignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOOpticalSquare: public OpticalObject
{

public:
  //---------- Constructors / Destructor
  OptOOpticalSquare(){ };
  OptOOpticalSquare(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data) : 
  OpticalObject( parent, type, name, copy_data){ };
  ~OptOOpticalSquare(){ };

  //---------- Fast simulation of deviation of the light ray (reflection, shift, ...)
  virtual void fastDeviatesLightRay( LightRay& lightray );
  //---------- Detailed simulation of the light ray traversing
  virtual void fastTraversesLightRay( LightRay& lightray );
  //---------- Detailed simulation of deviation of the light ray (reflection, shift, ...)
  virtual void detailedDeviatesLightRay( LightRay& lightray );
  //---------- Fast simulation of the light ray traversing
  virtual void detailedTraversesLightRay( LightRay& lightray );

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif

 private:
  //---------- Calculate the centre points and normal std::vector of each of the four pentaprism faces the light ray may touch
  void calculateFaces( ALIbool isDetailed );

  //---------- Centre points and normal std::vector of each of the four pentaprism faces the light ray may touch
  Hep3Vector faceP[5];
  Hep3Vector faceV[5]; 


};

#endif

