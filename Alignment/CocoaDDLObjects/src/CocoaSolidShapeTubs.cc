//   COCOA class implementation file
// Id:  CocoaSolidShapeTubs.cc
// CAT: Model
//
//   History: v1.0
//   Pedro Arce
#include <fstream>
#include <map>

#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"

CocoaSolidShapeTubs::CocoaSolidShapeTubs(
    const ALIstring type, ALIfloat pRMin, ALIfloat pRMax, ALIfloat pDz, ALIfloat pSPhi, ALIfloat pDPhi)
    : CocoaSolidShape(type) {
  theInnerRadius = pRMin;
  theOuterRadius = pRMax;
  theZHalfLength = pDz;
  theStartPhiAngle = pSPhi;
  theDeltaPhiAngle = pDPhi;
}
