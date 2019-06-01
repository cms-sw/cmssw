//   COCOA class header file
// Id:  CocoaSolidShapeTubs.h
//
//   History: v1.0
//   Pedro Arce

#ifndef _CocoaSolidShapeTubs_HH
#define _CocoaSolidShapeTubs_HH

#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShape.h"
#include "Alignment/CocoaUtilities/interface/CocoaGlobals.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

class CocoaSolidShapeTubs : public CocoaSolidShape {
public:
  //---------- Constructors / Destructor
  CocoaSolidShapeTubs(const ALIstring pType,
                      ALIfloat pRMin,
                      ALIfloat pRMax,
                      ALIfloat pDz,
                      ALIfloat pSPhi = 0. * deg,
                      ALIfloat pDPhi = 360. * deg);
  ~CocoaSolidShapeTubs() override{};
  ALIfloat getInnerRadius() const { return theInnerRadius; }
  ALIfloat getOuterRadius() const { return theOuterRadius; }
  ALIfloat getZHalfLength() const { return theZHalfLength; }
  ALIfloat getStartPhiAngle() const { return theStartPhiAngle; }
  ALIfloat getDeltaPhiAngle() const { return theDeltaPhiAngle; }

private:
  ALIfloat theInnerRadius;
  ALIfloat theOuterRadius;
  ALIfloat theZHalfLength;
  ALIfloat theStartPhiAngle;
  ALIfloat theDeltaPhiAngle;
};

#endif
