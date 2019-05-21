//   COCOA class implementation file
//Id:  OptOLaser.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOLaser.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ default behaviour: create a LightRay object
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOLaser::defaultBehaviour(LightRay& lightray, Measurement& meas) {
  if (ALIUtils::debug >= 3)
    std::cout << "create laser lightray " << std::endl;
  lightray.startLightRay(this);
}

#ifdef COCOA_VIS
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void OptOLaser::fillVRML() {
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  ALIColour* col = new ALIColour(1., 0., 0., 0.);
  vrmlmgr.AddBox(*this, 0.2, 0.2, 0.5, col);
  vrmlmgr.SendReferenceFrame(*this, 0.12);
  vrmlmgr.SendName(*this, 0.1);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOLaser::fillIguana() {
  ALIColour* col = new ALIColour(1., 0., 0., 0.);
  std::vector<ALIdouble> spar;
  spar.push_back(1.);
  spar.push_back(5.);
  CLHEP::HepRotation rm;
  rm.rotateX(90. * deg);
  IgCocoaFileMgr::getInstance().addSolid(*this, "CYLINDER", spar, col, CLHEP::Hep3Vector(), rm);
}
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOLaser::constructSolidShape() {
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go);

  theSolidShape = new CocoaSolidShapeTubs(
      "Tubs", go * 0. * cm / m, go * 1. * cm / m, go * 5. * cm / m);  //COCOA internal units are meters
}
