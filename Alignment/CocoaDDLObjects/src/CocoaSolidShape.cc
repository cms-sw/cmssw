//   COCOA class implementation file
// Id:  CocoaSolidShape.cc
// CAT: Model
//
//   History: v1.0
//   Pedro Arce
#include <fstream>
#include <map>
#include <utility>

#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShape.h"

CocoaSolidShape::CocoaSolidShape(ALIstring type) { theType = std::move(type); }
