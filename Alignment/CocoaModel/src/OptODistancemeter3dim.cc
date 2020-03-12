//   COCOA class implementation file
//Id:  OptODistancemeter3dim.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptODistancemeter3dim.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include <iostream>
#include <iomanip>
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Default behaviour: make measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter3dim::defaultBehaviour(LightRay& lightray, Measurement& meas) { makeMeasurement(lightray, meas); }

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Make measurement as distance to previous object 'screen'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter3dim::makeMeasurement(LightRay& lightray, Measurement& meas) {
  const OpticalObject* opto_prev = meas.getPreviousOptO(this);
  CLHEP::Hep3Vector dm_centre = centreGlob();
  dm_centre -= opto_prev->centreGlob();
  if (ALIUtils::debug >= 4) {
    std::cout << "OptO " << name() << std::endl;
    ALIUtils::dump3v(centreGlob(), " centre Glob ");
    std::cout << "OptO " << opto_prev->name() << std::endl;
    ALIUtils::dump3v(opto_prev->centreGlob(), " previous OptO centre Glob ");
    ALIUtils::dump3v(dm_centre, " distance std::vector");
  }

  meas.setValueSimulated(0, dm_centre.mag());
  if (ALIUtils::debug >= 1) {
    std::cout << "SIMU value: D: " << meas.valueSimulated(0) * 1000. << " (mm)  " << (this)->name() << std::endl;
    std::cout << "REAL value: D: " << meas.value()[0] * 1000. << " (mm)  " << (this)->name() << std::endl;
  }
}

#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter3dim::fillVRML() {
  //-  std::cout << " filling optosensor " << std::endl;
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  vrmlmgr.AddBox(*this, 1., .2, .2);
  vrmlmgr.SendReferenceFrame(*this, 1.);
  vrmlmgr.SendName(*this, 0.01);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter3dim::fillIguana() {
  ALIColour* col = new ALIColour(1., 0., 1., 0.);
  std::vector<ALIdouble> spar;
  spar.push_back(1.);
  spar.push_back(3.);
  CLHEP::HepRotation rm;
  rm.rotateX(90. * deg);
  IgCocoaFileMgr::getInstance().addSolid(*this, "CYLINDER", spar, col, CLHEP::Hep3Vector(), rm);
}
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter3dim::constructSolidShape() {
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go);

  theSolidShape = new CocoaSolidShapeTubs(
      "Tubs", go * 0. * cm / m, go * 2. * cm / m, go * 5. * cm / m);  //COCOA internal units are meters
}
