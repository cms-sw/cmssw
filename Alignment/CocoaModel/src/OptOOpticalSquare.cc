//   COCOA class implementation file
//Id:  OptOOpticalSquare
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOOpticalSquare.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include <iostream>
#include <iomanip>
#ifdef COCOA_VIS
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/CocoaVisMgr/interface/ALIColour.h"
#endif
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

using namespace CLHEP;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Detailed simulation of deviation of the light ray 
//@@  Refract at entering, reflect in two of the plates and gets deviated 90o, refract at exiting
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOOpticalSquare::detailedDeviatesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED DEVIATION IN OPTICAL SQUARE " << name() << std::endl;

  calculateFaces( true );

  //---------- Deviate in prism
  //---------- Refract after entering face 0, reflect in face 1, reflect in face 2, refract after face 3
  const ALIdouble refra_ind = findExtraEntryValueMustExist("refra_ind");
  const ALIdouble refra_ind0 = 1.; 

  ALIint ii;
  for( ii = 0; ii < 4; ii++ ) {
    if(ii == 0) { 
      if( ALIUtils::debug >= 3 ) std::cout << "## OPTOOPTICALSQUARE: refract in face " << ii << std::endl;
      lightray.refract( ALIPlane(faceP[ii],faceV[ii]), refra_ind0, refra_ind );
    } else if( ii == 3) { 
    //---- interchange refraction index for exiting instead of entering
      lightray.refract( ALIPlane(faceP[ii],faceV[ii]), refra_ind, refra_ind0 );
    } else { 
      lightray.reflect( ALIPlane(faceP[ii],faceV[ii] ));
    }  
    if (ALIUtils::debug >= 3) { 
      lightray.dumpData( "After face ");
    } 
  }

  //----- checks that it is inside prism and the order of faces hit is the good one
  //   

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Detailed simulation of traversing of the light ray 
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOOpticalSquare::detailedTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED TRAVERSES OPTICAL SQUARE " << name() << std::endl;

  calculateFaces( true );
  const ALIdouble refra_ind = findExtraEntryValueMustExist("refra_ind");
  const ALIdouble refra_ind0 = 1.; 

  lightray.refract( ALIPlane(faceP[0],faceV[0]), refra_ind0, refra_ind );
  lightray.refract( ALIPlane(faceP[4],faceV[4]), refra_ind, refra_ind0 );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of deviation of the light ray
//@@ Reflect in two of the plates and gets deviated 90o
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOOpticalSquare::fastDeviatesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST DEVIATION IN OPTICAL SQUARE " << name() << std::endl;
  
  calculateFaces( false );
  
  //---------- Deviate in prism
  lightray.reflect( ALIPlane(faceP[1],faceV[1] ));
  if (ALIUtils::debug >= 3) { 
    lightray.dumpData( "After face 1");
  } 
  lightray.reflect( ALIPlane(faceP[2],faceV[2] ));
  if (ALIUtils::debug >= 3) { 
    lightray.dumpData( "After face 2");
  } 
  lightray.intersect( ALIPlane(faceP[3],faceV[3] ));
  if (ALIUtils::debug >= 3) { 
    lightray.dumpData( "intersected at face 3");
  }

  //----- Deviates by 'devi' X & Y??
  lightray.shiftAndDeviateWhileTraversing( this, 'R');
  /*  ALIdouble deviRX = findExtraEntryValue("deviRX");
  ALIdouble deviRY = findExtraEntryValue("deviRY");
  ALIdouble shiftRX = findExtraEntryValue("shiftRX");
  ALIdouble shiftRY = findExtraEntryValue("shiftRY");
  lightray.shiftAndDeviateWhileTraversing( this, shiftRX, shiftRY, deviRX, deviRY);
  */

  if (ALIUtils::debug >= 2) {
    //  std::cout << " shiftRX " << shiftRX << " shiftRY " << shiftRY << std::endl;
    // std::cout << " deviRX " << deviRX << " deviRY " << deviRY << std::endl;
    lightray.dumpData("Deviated ");
  }


}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ 
//@@ 
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  //---------- Fast simulation of the light ray traversing
void OptOOpticalSquare::fastTraversesLightRay( LightRay& lightray )
{
  //  std::cerr << " WARNING  there should be an extra piece to make entering and exiting surfaces parallel (like in modified_rhomboid_prism) " << std::endl;

  calculateFaces( false );

  lightray.intersect( ALIPlane(faceP[1],faceV[1] ));
  //---------- Shift and Deviate
  lightray.shiftAndDeviateWhileTraversing( this, 'T');
  /*  ALIdouble shiftX = findExtraEntryValue("shiftTX");
  ALIdouble shiftY = findExtraEntryValue("shiftTY");
  ALIdouble deviTX = findExtraEntryValue("deviTX");
  ALIdouble deviTY = findExtraEntryValue("deviTY");
  lightray.shiftAndDeviateWhileTraversing( this, shiftX, shiftY, deviTX, deviTY);
  */
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Shifted and Deviated");
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate the centre points and normal std::vector of each of the four pentaprism faces the light ray may touch
//@@ Build the four faces. 0: entry, 1: 1st reflection, 2: 2nd reflection, 3: exit  (look at figure in documentation)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOOpticalSquare::calculateFaces( ALIbool isDetailed )
{

  ALIint numberOfFaces = 5;

  //----- useful variables
  CLHEP::Hep3Vector opto_centre = centreGlob();
  if(ALIUtils::debug >= 3) std::cout << "opto_centre " << opto_centre << std::endl;
  const ALIdouble hlen1 = findExtraEntryValueMustExist("length1") / 2.;
  const ALIdouble hlen2 = findExtraEntryValueMustExist("length2") / 2.;
  ALIdouble ang = 67.5*acos(0.)/90.;

  //-  if( ALIUtils::debug >= 3) std::cout << "length 1 " << 2*hlen1 << " length 2 " << hlen2 * 2 << std::endl;
  faceP[0] = CLHEP::Hep3Vector( 0, 0 , -hlen1 );
  faceV[0] = CLHEP::Hep3Vector( 0, 0, -1 );
  faceP[1] = CLHEP::Hep3Vector( 0, hlen1 - hlen2*sin(ang), hlen1 + hlen2*cos(ang) );
  faceV[1] = CLHEP::Hep3Vector( 0, cos(ang), sin(ang) );

  faceP[2] = CLHEP::Hep3Vector( 0, -hlen1 - hlen2*cos(ang), -hlen1 + hlen2*sin(ang) );
  faceV[2] = CLHEP::Hep3Vector( 0, -sin(ang), -cos(ang) );
  faceP[3] = CLHEP::Hep3Vector( 0, hlen1, 0 );
  faceV[3] = CLHEP::Hep3Vector( 0, 1, 0 );

  // face of added piece (so that light when traversing finds parallel surfaces at entry and exit)
  faceP[4] = CLHEP::Hep3Vector( 0, 0, hlen1 + 2*hlen2*cos(ang) );
  faceV[4] = CLHEP::Hep3Vector( 0, 0, 1 );

  //--------- Put faces in global reference frame
  CLHEP::HepRotation rmt = rmGlob();
  ALIint ii;
  if (ALIUtils::debug >= 3) { 
    std::cout << " optical_square centre" << opto_centre << std::endl;
  }
  for( ii = 0; ii < numberOfFaces; ii++ ) {
    faceP[ii] = rmt * faceP[ii];
    faceP[ii] += opto_centre;
    faceV[ii] = rmt * faceV[ii];
    if (ALIUtils::debug >= 3) { 
      std::cout << "point at face " << ii << ": " << faceP[ii] << std::endl;
      std::cout << "normal at face " << ii << ": " << faceV[ii] << std::endl;
    }
  }

  //----------- Correct faces 1 & 2 by wedge: rotate each face normal 1/2 of the wedge around two axis perpendicular to normal 
  if( isDetailed ) {
    ALIdouble wedge, wedgeX, wedgeY;
    const ALIbool wxy = findExtraEntryValueIfExists("wedge", wedge);
    if( !wxy ) {
      wedgeX = findExtraEntryValue("wedgeX");
      wedgeY = findExtraEntryValue("wedgeY");
    } else {
      wedgeX = wedge;
      wedgeY = wedge;
    }

    //----- One axis is along X axis for the two faces (X belong to both faces)
    if(ALIUtils::debug >= 4) std::cout << "OptOOpticalSquare calculateFaces: wedgeX  " << wedgeX << " wedgeY  " << wedgeY << std::endl;
    CLHEP::Hep3Vector Axis1(1.,0.,0.);
    Axis1 = rmt * Axis1; 
    if( ALIUtils::debug >= 4) {
      ALIUtils::dump3v(faceV[1], "faceV[1] before wedge"); 
      ALIUtils::dump3v(faceV[2], "faceV[2] before wedge"); 
    }
    faceV[1].rotate(0.5* wedgeX, Axis1);
    if( ALIUtils::debug >= 4) ALIUtils::dump3v( Axis1, " Axis1 in faceV[1] ");
    faceV[2].rotate(-0.5* wedgeX, Axis1);
    if( ALIUtils::debug >= 4) {
      ALIUtils::dump3v( Axis1, " Axis1 in faceV[2] ");
      ALIUtils::dump3v(faceV[1], "faceV[1] after wedge X"); 
      ALIUtils::dump3v(faceV[2], "faceV[2] after wedge X"); 
    }

    //----- Other axis perpendicular to first and to normal of each face
    CLHEP::Hep3Vector Axis2 = Axis1;
    Axis2 = Axis2.cross( faceV[1] );
    faceV[1].rotate(0.5* wedgeY, Axis2);
    if( ALIUtils::debug >= 4) ALIUtils::dump3v( Axis2, " Axis2 in faceV[1] ");
    Axis2 = Axis1;
    Axis2 = Axis2.cross( faceV[2] );
    faceV[2].rotate(-0.5* wedgeY, Axis2);
    if( ALIUtils::debug >= 4) {
      ALIUtils::dump3v( Axis2, " Axis2 in faceV[2] ");
      ALIUtils::dump3v(faceV[1], "faceV[1] after wedge Y"); 
      ALIUtils::dump3v(faceV[2], "faceV[2] after wedge Y"); 
    }
    
  }

}


#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOOpticalSquare::fillIguana()
{
  ALIColour* col = new ALIColour( 0., 0., 1., 0. );
  ALIdouble length1;
  ALIbool wexists = findExtraEntryValueIfExists("length1",length1);
  if( !wexists ) length1 = 4.;
  ALIdouble length2;
  wexists = findExtraEntryValueIfExists("length2",length2);
  if( !wexists ) length2 = 4.;

  std::vector<ALIdouble> spar;
  spar.push_back(length1);
  spar.push_back(length2);
  IgCocoaFileMgr::getInstance().addSolid( *this, "OPTSQR", spar, col);
}
#endif


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOOpticalSquare::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*5.*cm/m, go*5.*cm/m, go*5.*cm/m ); //COCOA internal units are meters
}
