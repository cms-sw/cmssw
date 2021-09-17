//   COCOA class implementation file
//Id:  ALIRmDataFromFile.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/ALIRmDataFromFile.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIRmDataFromFile::ALIRmDataFromFile() {}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool ALIRmDataFromFile::setAngle(const ALIstring& coord, const ALIdouble val) {
  if (coord == "X") {
    return setAngleX(val);
  } else if (coord == "Y") {
    return setAngleY(val);
  } else if (coord == "Z") {
    return setAngleZ(val);
  } else {
    std::cerr << "!!! FATAL ERROR ALIRmDataFromFile::setAngle. Coordinate must be X, Y or Z, it ii " << coord
              << std::endl;
    std::exception();
  }
  return false;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool ALIRmDataFromFile::setAngleX(const ALIdouble val) {
  theAngleX = val;
  theDataFilled += "X";
  return true;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool ALIRmDataFromFile::setAngleY(const ALIdouble val) {
  theAngleY = val;
  theDataFilled += "Y";
  return true;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool ALIRmDataFromFile::setAngleZ(const ALIdouble val) {
  theAngleZ = val;
  theDataFilled += "Z";
  return true;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void ALIRmDataFromFile::constructRm() {
  if (theDataFilled.find('X') == std::string::npos || theDataFilled.find('Y') == std::string::npos ||
      theDataFilled.find('Z') == std::string::npos) {
    std::cerr << "!!!  ALIRmDataFromFile::constructRm. FATAL ERROR: building rm while one angle is missing: "
              << theDataFilled << std::endl;
  } else {
    theRm = CLHEP::HepRotation();
    theRm.rotateX(theAngleX);
    theRm.rotateY(theAngleY);
    theRm.rotateZ(theAngleZ);
  }
}
