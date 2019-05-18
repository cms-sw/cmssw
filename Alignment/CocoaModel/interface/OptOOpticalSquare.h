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

#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
class Measurement;
class LightRay;

class OptOOpticalSquare : public OpticalObject {
public:
  //---------- Constructors / Destructor
  OptOOpticalSquare(){};
  OptOOpticalSquare(OpticalObject* parent, const ALIstring& type, const ALIstring& name, const ALIbool copy_data)
      : OpticalObject(parent, type, name, copy_data){};
  ~OptOOpticalSquare() override{};

  //---------- Fast simulation of deviation of the light ray (reflection, shift, ...)
  void fastDeviatesLightRay(LightRay& lightray) override;
  //---------- Detailed simulation of the light ray traversing
  void fastTraversesLightRay(LightRay& lightray) override;
  //---------- Detailed simulation of deviation of the light ray (reflection, shift, ...)
  void detailedDeviatesLightRay(LightRay& lightray) override;
  //---------- Fast simulation of the light ray traversing
  void detailedTraversesLightRay(LightRay& lightray) override;

#ifdef COCOA_VIS
  virtual void fillIguana();
#endif
  void constructSolidShape() override;

private:
  //---------- Calculate the centre points and normal std::vector of each of the four pentaprism faces the light ray may touch
  void calculateFaces(ALIbool isDetailed);

  //---------- Centre points and normal std::vector of each of the four pentaprism faces the light ray may touch
  CLHEP::Hep3Vector faceP[5];
  CLHEP::Hep3Vector faceV[5];
};

#endif
