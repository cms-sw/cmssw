//   COCOA class implementation file
//Id:  OptOPinhole.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "OpticalAlignment/CocoaModel/interface/OptOPinhole.h"
#include "OpticalAlignment/CocoaModel/interface/LightRay.h"
#include "OpticalAlignment/CocoaModel/interface/Measurement.h"
#include "OpticalAlignment/CocoaModel/interface/Model.h"
#include <iostream>
#include <iomanip>
#ifdef COCOA_VIS
#include "OpticalAlignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "OpticalAlignment/CocoaVisMgr/interface/ALIColour.h"
#endif
#include "CLHEP/Units/SystemOfUnits.h"


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Traverse pinhole
//@@ 1. the lightray direction is changed to the one that makes the ray traverse the pinhole
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOPinhole::defaultBehaviour( LightRay& lightray, Measurement& meas )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: TRAVERSE PINHOLE  " << name() << std::endl;
  
  //----------- Direction is the one that links the source and the pinhole
  CLHEP::Hep3Vector source = lightray.point();
  CLHEP::Hep3Vector pinhole = centreGlob();
  lightray.setDirection( pinhole - source );
  lightray.setPoint( pinhole );

  if (ALIUtils::debug >= 4) {
    ALIUtils::dump3v( source, " source centre ");
    ALIUtils::dump3v( pinhole, " pinhole centre ");
  }
  if (ALIUtils::debug >= 3) {
    lightray.dumpData( "lightray at pinhole ");
  }

}


#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOPinhole::fillIguana()
{
  ALIColour* col = new ALIColour( 1., 1., 1., 0. );
  std::vector<ALIdouble> spar;
  spar.push_back(0.2);
  spar.push_back(0.5);
  CLHEP::HepRotation rm;
  rm.rotateX( 90.*deg);
  IgCocoaFileMgr::getInstance().addSolid( *this, "CYLINDER", spar, col, CLHEP::Hep3Vector(), rm);
}

#endif
