//   COCOA class implementation file
//Id:  OptODistancemeter.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptODistancemeter.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h"
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
void OptODistancemeter::defaultBehaviour(LightRay& lightray, Measurement& meas) { makeMeasurement(lightray, meas); }

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Make measurement as distance to previous object 'screen'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter::makeMeasurement(LightRay& lightray, Measurement& meas) {
  const OpticalObject* opto_prev = meas.getPreviousOptO(this);
  CLHEP::Hep3Vector dm_centre = centreGlob();
  CLHEP::Hep3Vector ZAxis(0., 0., 1.);
  CLHEP::HepRotation rmt = rmGlob();
  ZAxis = rmt * ZAxis;

  //----- intersect z of distancemeter with distance target
  ALILine dm_line(centreGlob(), ZAxis);

  CLHEP::Hep3Vector ZAxisdt(0., 0., 1.);
  const CLHEP::HepRotation& rmtdt = opto_prev->rmGlob();
  ZAxisdt = rmtdt * ZAxisdt;
  ALIPlane dt_plane(opto_prev->centreGlob(), ZAxisdt);

  if (ALIUtils::debug >= 3) {
    ALIUtils::dump3v(dm_centre, "distancemeter centre");
    ALIUtils::dump3v(ZAxis, "distancemeter direc");
    ALIUtils::dump3v(opto_prev->centreGlob(), "distance_target centre");
    ALIUtils::dump3v(ZAxisdt, "distance_target direc");
  }

  CLHEP::Hep3Vector inters = dm_line.intersect(dt_plane);

  dm_centre = inters - dm_centre;
  if (ALIUtils::debug >= 4) {
    std::cout << "OptO dm" << name() << dm_line << std::endl;
    //    ALIUtils::dump3v( centreGlob(), " centre Glob ");
    std::cout << "OptO dt" << opto_prev->name() << std::endl;
    ALIUtils::dump3v(opto_prev->centreGlob(), " previous OptO centre Glob ");
    ALIUtils::dump3v(inters, " intersection with target ");
    ALIUtils::dump3v(dm_centre, " distance std::vector");
  }
  ALIdouble proj = dm_centre * ZAxis;

  //- ALIUtils::dump3v( ZAxis, " zaxis ");
  //-  std::cout << " proj " << proj << std::endl;

  meas.setValueSimulated(0, proj);
  if (ALIUtils::debug >= 1) {
    std::cout << "SIMU value: D: " << meas.valueSimulated(0) * 1000. << " (mm)  " << (this)->name() << std::endl;
    std::cout << "REAL value: D: " << meas.value()[0] * 1000. << " (mm)  " << (this)->name() << std::endl;
  }
}

#ifdef COCOA_VIS
void OptODistancemeter::fillVRML() {
  //-  std::cout << " filling optosensor " << std::endl;
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  ALIColour* col = new ALIColour(1., 0., 1., 0.);
  vrmlmgr.AddBox(*this, .2, .2, 1., col);
  vrmlmgr.SendReferenceFrame(*this, 1.2);
  vrmlmgr.SendName(*this, 0.1);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter::fillIguana() {
  ALIColour* col = new ALIColour(1., 0., 0.5, 0.);
  std::vector<ALIdouble> spar;
  spar.push_back(1.);
  spar.push_back(3.);
  CLHEP::HepRotation rm;
  rm.rotateX(90. * deg);
  IgCocoaFileMgr::getInstance().addSolid(*this, "CYLINDER", spar, col, CLHEP::Hep3Vector(), rm);
}
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptODistancemeter::constructSolidShape() {
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go);

  theSolidShape = new CocoaSolidShapeTubs(
      "Tubs", go * 0. * cm / m, go * 2. * cm / m, go * 5. * cm / m);  //COCOA internal units are meters
}
