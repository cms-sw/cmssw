//   COCOA class implementation file
// Id:  CocoaSolidShapeBox.cc
// CAT: Model
//
//   History: v1.0
//   Pedro Arce
#include <fstream>
#include <map>

#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"

CocoaSolidShapeBox::CocoaSolidShapeBox(ALIstring type, ALIfloat xdim, ALIfloat ydim, ALIfloat zdim)
    : CocoaSolidShape(type) {
  theXHalfLength = xdim;
  theYHalfLength = ydim;
  theZHalfLength = zdim;
}
