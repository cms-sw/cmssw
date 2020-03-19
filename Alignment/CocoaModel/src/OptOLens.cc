//   COCOA class implementation file
//Id:  OptOLens.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOLens.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#ifdef COCOA_VIS
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/CocoaVisMgr/interface/ALIColour.h"
#endif
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OptOLens::OptOLens() {
  setRmGlobalOriginal(CLHEP::HepRotation());
  setRmGlobal(CLHEP::HepRotation());
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOLens::participateInMeasurement(LightRay& lightray, Measurement& meas, const ALIstring& behav) {
  std::cerr << "object not implemented yet " << std::endl;
  exit(1);
}

#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOLens::fillIguana() {
  ALIColour* col = new ALIColour(0.5, 1., 0.5, 0.);
  std::vector<ALIdouble> spar;
  spar.push_back(1.);
  spar.push_back(0.1);
  CLHEP::HepRotation rm;
  rm.rotateX(90. * deg);
  IgCocoaFileMgr::getInstance().addSolid(*this, "CYLINDER", spar, col, CLHEP::Hep3Vector(), rm);
}
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOLens::constructSolidShape() {
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go);

  theSolidShape = new CocoaSolidShapeTubs(
      "Tubs", go * 0. * cm / m, go * 5. * cm / m, go * 1. * cm / m);  //COCOA internal units are meters
}
