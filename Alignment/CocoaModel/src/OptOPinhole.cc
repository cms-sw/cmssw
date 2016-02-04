//   COCOA class implementation file
//Id:  OptOPinhole.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOPinhole.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include <iostream>
#include <iomanip>
#ifdef COCOA_VIS
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/CocoaVisMgr/interface/ALIColour.h"
#endif
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"


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


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOPinhole::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeTubs( "Tubs", go*0.*cm/m, go*1.*cm/m, go*1.*cm/m ); //COCOA internal units are meters
}
