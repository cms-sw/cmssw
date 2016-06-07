//   COCOA class implementation file
//Id:  OptOUserDefined.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOUserDefined.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/ALIVisLightPath.h"
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Default behaviour: make measurement
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOUserDefined::userDefinedBehaviour( LightRay& lightray, Measurement& meas, const ALIstring& behav )
{
#ifdef COCOA_VIS
  ALIVisLightPath* vispath = 0;
  if( ALIUtils::getFirstTime() ) {
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if(gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
      vispath = IgCocoaFileMgr::getInstance().newLightPath( name() );
    }
  }
#endif

  if(ALIUtils::debug >= 5)ALIUtils::dump3v( centreGlob(), " user Defined centre " );
  //---- Object is not a basic one
  if(ExtraEntryList().size() == 0) {
    std::cerr << "OpticalObject: !!! EXITING at Measurement: " << meas.name() << " in OptO: "  << name() << " behaviour ':" << behav << "' is not adequate " << std::endl;
    std::cerr <<  "an OptO has to indicate if detailed simulation traverses (:T) or deviates (:D) or Fast Simulation traverses (:FT) or deviates  (:FD) or default behaviour () "<< std::endl;
    exit(4);
  } else {
    ALIint behavSize = behav.size();
    //     if( !(nSet[0]).isNumber() ) nSet = "";
    ALIstring nSet;
    if( behavSize != 2 || behav[0] != 'D' ){  //|| !behav[1]).isNumber() )
      std::cerr << "OpticalObject: !!! EXITING at Measurement: " << meas.name() << " in OptO: "  << name() << " behaviour ':" << behav << "' is not adequate " << std::endl;
      std::cerr <<  "an OptO has to indicate detailed simulation by (:Dn) where n is an integer number " << std::endl;
      exit(4);
    } else {
      nSet = behav.substr( behavSize-1, behavSize);
    }
    //-    std::cout << behavSize << " nSet " << nSet << std::endl;
    ALIdouble shiftZ = findExtraEntryValue("shiftZ"+nSet);
    ALIdouble shiftX = findExtraEntryValue("shiftX"+nSet);
    ALIdouble shiftY = findExtraEntryValue("shiftY"+nSet);
    ALIdouble deviX = findExtraEntryValue("deviX"+nSet);
    ALIdouble deviY = findExtraEntryValue("deviY"+nSet);
    ALIdouble deviZ = findExtraEntryValue("deviZ"+nSet);
    CLHEP::Hep3Vector shift3D( shiftX, shiftY, shiftZ );
    CLHEP::HepRotation rmt = rmGlob();
    shift3D = rmt*shift3D;
    if(ALIUtils::debug >= 5) {
      lightray.dumpData("OptOUserDefined: lightray incoming");
      ALIUtils::dump3v( shift3D, " shift 3D " );
                    //-std::cout << " shift " << shiftX << " shiftY " << shiftY << " shiftZ " << shiftZ
                    //-    << " deviX " << deviX << " deviY " << deviY << std::endl;
    }

    ALIPlane plate = getPlate(0, 0);
    lightray.intersect( plate );

#ifdef COCOA_VIS
    //--- draw a point at intersection
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if( ALIUtils::getFirstTime() ) {
      if(gomgr->GlobalOptions()["VisWriteVRML"] > 1) {
        ALIVRMLMgr::getInstance().addLightPoint( lightray.point() );
        if(ALIUtils::debug>=5)std::cout << "ALIVRMLMgr addLightPoint " << lightray.point() << name() << std::endl;
      }
      if(gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
        vispath->addLightPoint( lightray.point(), this );
      }
    }
#endif

    lightray.setPoint( lightray.point() + shift3D );
    if(ALIUtils::debug >= 5) {
      lightray.dumpData("OptOUserDefined: lightray after shift");
    }
    CLHEP::Hep3Vector direc = lightray.direction();
    CLHEP::Hep3Vector XAxis(1.,0.,0.);
    XAxis = rmt*XAxis;
    direc.rotate(deviX, XAxis);
    if(ALIUtils::debug >= 5) {
      std::cout << "Direction after deviX " << direc << std::endl;
      std::cout << " deviX " << deviX << std::endl;
    }
    CLHEP::Hep3Vector YAxis(0.,1.,0.);
    YAxis = rmt*YAxis;
    direc.rotate(deviY, YAxis);
    lightray.setDirection( direc );
    if(ALIUtils::debug >= 5) {
      std::cout << "Direction after deviY " << direc << std::endl;
      std::cout << " deviY " << deviY << std::endl;
    }
    CLHEP::Hep3Vector ZAxis(0.,0.,1.);
    ZAxis = rmt*ZAxis;
    direc.rotate(deviZ, ZAxis);
    lightray.setDirection( direc );
    if(ALIUtils::debug >= 5) {
      std::cout << "Direction after deviZ " << direc << std::endl;
      std::cout << " deviZ " << deviZ << std::endl;
    }
    if(ALIUtils::debug >= 4) {
      lightray.dumpData("OptOUserDefined: lightray at exiting");
    }
  }

#ifdef COCOA_VIS
  //--- draw a point at exiting
  if( ALIUtils::getFirstTime() ) {
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if(gomgr->GlobalOptions()["VisWriteVRML"] > 1) {
      ALIVRMLMgr::getInstance().addLightPoint( lightray.point() );
      if(ALIUtils::debug>=5)std::cout << "ALIVRMLMg  addLightPoint " << lightray.point() << name() << std::endl;
    }
    if(gomgr->GlobalOptions()["VisWriteIguana"] > 1) {
      vispath->addLightPoint( lightray.point(), this );
    }
  }
#endif

}

#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOUserDefined::fillVRML()
{
  /*  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  ALIColour* col = new ALIColour( 1., 0.7, 0.8, 0. );
  vrmlmgr.AddBox( *this, 100, 100, 0.1, col);
  vrmlmgr.SendReferenceFrame( *this, 0.1);
  vrmlmgr.SendName( *this, 0.01 );
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOUserDefined::fillIguana()
{
  ALIColour* col = new ALIColour( 0., 0., 0., 0. );
  std::vector<ALIdouble> spar;
  spar.push_back(1.);
  spar.push_back(1.);
  spar.push_back(1.);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, col);
}
#endif

