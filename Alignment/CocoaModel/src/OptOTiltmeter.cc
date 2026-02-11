//   COCOA class implementation file
//Id:  OptOTiltmeter.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOTiltmeter.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include <iostream>
#include <iomanip>
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

using namespace CLHEP;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Default behaviour: make measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOTiltmeter::defaultBehaviour(LightRay& lightray, Measurement& meas) { makeMeasurement(lightray, meas); }

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Make measurement as angle with the horizontal plane = angle between local  Z axis and its projection on the global XZ plane
//-Make measurement as rotations around X axis: difference between current Z axis and Z axis (0,0,1)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOTiltmeter::makeMeasurement(LightRay& lightray, Measurement& meas) {
  //---------- Get local Z axix
  CLHEP::HepRotation rmt = rmGlob();
  CLHEP::Hep3Vector ZAxisl(0., 0., 1.);
  ZAxisl *= rmt;

  //--------- Get projection in a global XZ plane
  /*-plane parallel to global Y (gravity) and to tiltmeter Z
  CLHEP::Hep3Vector plane_point(0.,0.,0.);
  CLHEP::Hep3Vector plane_normal = ZAxisl.cross( CLHEP::Hep3Vector(0.,1.,0.) );
  CLHEP::Hep3Vector ZAxis_proj = (ALIPlane( plane_point, plane_normal)).project( ZAxisl );
  */
  CLHEP::Hep3Vector XAxisg(1., 0., 0.);
  CLHEP::Hep3Vector ZAxisg(0., 0., 1.);
  CLHEP::Hep3Vector ZAxis_proj = (ZAxisl * XAxisg) * XAxisg + (ZAxisl * ZAxisg) * ZAxisg;
  ZAxis_proj *= (1. / ZAxis_proj.mag());

  //--------- Get measurement
  ALIdouble measvalue = acos(ZAxisl * ZAxis_proj / ZAxisl.mag() / ZAxis_proj.mag());
  //----- get sign of angle as sign of y component of ZAxis
  if (ZAxisl.y() != 0)
    measvalue *= (ZAxisl.y() / fabs(ZAxisl.y()));
  meas.setValueSimulated(0, measvalue);

  if (ALIUtils::debug >= 3) {
    ALIUtils::dump3v(ZAxisl, " OptOTiltmeter: Local Z axis ");
    ALIUtils::dumprm(rmt, " tiltmeter rotation matrix");
    ALIUtils::dump3v(ZAxis_proj, " Projection of Local Z axis on global XZ plane ");
    std::cout << "SIMU value: TA: " << std::setprecision(8) << 1000 * meas.valueSimulated(0) << " (mrad)  "
              << (this)->name() << std::endl;
    std::cout << "REAL value: TA: " << std::setprecision(8) << 1000 * meas.value()[0] << " (mrad)  " << (this)->name()
              << std::endl;
  }

  /*-  //---------- Get simulated value:
  CLHEP::HepRotation rmtori = rmGlobOriginal();
  CLHEP::Hep3Vector ZAxism(0.,0.,1.);
  CLHEP::Hep3Vector ZAxism_ori = ZAxism;
  ZAxism_ori *= rmtori;

  //---------- Measure rotation with respect to original position, around the X axis defined by the original position, in the original YZ plane
  CLHEP::Hep3Vector ZAxism_rot = ZAxism;
  CLHEP::HepRotation rmt = rmGlob();
  ZAxism_rot *= rmt;
  //----- Project on original YZ plane
  CLHEP::Hep3Vector YAxism_ori(0.,1.,0.);
  YAxism_ori *= rmtori;
  //--- create original YZ plane
  CLHEP::Hep3Vector YZplanePoint = centreGlob();
  CLHEP::Hep3Vector YZplaneNormal = YAxism_ori.cross( ZAxism_ori );
  ALIPlane yzorig( YZplanePoint, YZplaneNormal );
  CLHEP::Hep3Vector ZAxism_proj = yzorig.project( ZAxism_rot);
  //- ALIUtils::dump3v( YAxism_ori, "YAxism_ori");
  //- ALIUtils::dump3v( ZAxism_ori, "ZAxism_ori");
  //- ALIUtils::dump3v( ZAxism_rot, "ZAxism_rot");
  //- ALIUtils::dump3v( ZAxism_proj, "ZAxism_proj");
  ALIdouble measValue =  acos( ZAxism.dot(ZAxism_proj)/ZAxism_proj.mag() );
  if( ZAxism_proj.x() < 0) measValue *= -1.;
  meas.setValueSimulated(0 , measValue );

  if (ALIUtils::debug >= 3) {
    std::cout << " OptOTiltmeter: Original Z axis " << ZAxism_ori << std::endl;
    ALIUtils::dumprm(rmt," tiltmeter original rotation matrix");
    std::cout << " OptOTiltmeter: current Z axis " << ZAxism_rot << std::endl;
    ALIUtils::dumprm(rmt," tiltmeter current rotation matrix");
    std::cout << "SIMU value; TA: " << std::setprecision(8) << meas.valueSimulated(0)
 	 << " (rad)  " << (this)->name() << std::endl;
    std::cout << "REAL value: TA: " << std::setprecision(8) << meas.value()[0] 
 	 << " (rad)  " << (this)->name() << std::endl;

  }
  */
}

#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOTiltmeter::fillVRML() {
  //-  std::cout << " filling optosensor " << std::endl;
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  ALIColour* col = new ALIColour(1., 1., 0., 0.);
  vrmlmgr.AddBox(*this, .2, .2, 1., col);
  vrmlmgr.SendReferenceFrame(*this, 0.6);
  vrmlmgr.SendName(*this, 0.01);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOTiltmeter::fillIguana() {
  ALIColour* col = new ALIColour(1., 0., 0.9, 0.);
  std::vector<ALIdouble> spar;
  spar.push_back(1.);
  spar.push_back(1.);
  spar.push_back(4.);
  IgCocoaFileMgr::getInstance().addSolid(*this, "BOX", spar, col);
}
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOTiltmeter::constructSolidShape() {
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go);

  theSolidShape = new CocoaSolidShapeBox(
      "Box", go * 2. * cm / m, go * 2. * cm / m, go * 5. * cm / m);  //COCOA internal units are meters
}
