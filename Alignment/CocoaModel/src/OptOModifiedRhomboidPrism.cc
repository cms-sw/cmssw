//  COCOA class implementation file
//Id:  OptOMOdifiedRhomboidPrism.cc
//CAT: Model
//
//   History: v0.9 Dec 1999 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOModifiedRhomboidPrism.h"
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

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of Reflection in Plate Splitter
//@@ The software gets the plane of reflection as the forward splitter plane
//@@ Then the beam is reflected in this plane.
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOModifiedRhomboidPrism::detailedDeviatesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "$$$$$ LR: DETAILED DEVIATION IN MODIFIED RHOMBOID PRISM " << name() << std::endl;

  CLHEP::Hep3Vector XAxis(1.,0.,0.);
  CLHEP::HepRotation rmt = rmGlob();
  XAxis = rmt*XAxis;
  CLHEP::Hep3Vector YAxis(0.,1.,0.);
  YAxis = rmt*YAxis;
  CLHEP::Hep3Vector ZAxis(0.,0.,1.);
  ZAxis = rmt*ZAxis;

  ALIUtils::dump3v( XAxis , " x axis ");
  ALIUtils::dump3v( YAxis , " y axis ");
  ALIUtils::dump3v( ZAxis , " z axis ");
  if (ALIUtils::debug >= 5) {
    ALIUtils::dump3v( centreGlob(), " centre ");
  }

  if (ALIUtils::debug >= 2) std::cout << "$$$ LR: REFRACTION IN FORWARD PLATE " << std::endl;
  //---------- Get forward plate
  ALIPlane plate = getPlate( true, true );
  //---------- Refract in plate while entering
  ALIdouble refra_ind1 = 1.;
  ALIdouble refra_ind2 = findExtraEntryValueMustExist("refra_ind");
  lightray.refract( plate, refra_ind1, refra_ind2 );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("LightRay after Refraction at entering: "); 
  }

  if (ALIUtils::debug >= 2) std::cout << std::endl << "$$$ LR: REFLECTION IN FIRST PLATE " << std::endl;
  //---------- Get up plate rotated
  plate = getRotatedPlate( true );
  //---------- Reflect in plate
  lightray.reflect( plate );

  if (ALIUtils::debug >= 2) std::cout << std::endl << "$$$ LR: REFLECTION IN SECOND PLATE " << std::endl;
  //---------- Get up plate rotated
  plate = getRotatedPlate( false );
  //---------- Reflect in plate
  lightray.reflect( plate );

  if (ALIUtils::debug >= 2) std::cout << std::endl << "$$$ LR: REFRACTION IN BACKWARD PLATE " << std::endl;
  //---------- Get backward plate
  plate = getPlate( false, true );
  //---------- Refract in plate while exiting
  lightray.refract( plate, refra_ind2, refra_ind1 );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of the light ray traversing
//@@  The beam enters the prism, is refracted, traverses the prism and finally is again refracted when it exits:
//@@ Get the intersection with the forward prism plane
//@@ Refract the beam and propagate until it intersects the backward plane.
//@@ Finally the beam is refracted again.
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOModifiedRhomboidPrism::detailedTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED TRAVERSE MODIFIED RHOMBOID PRISM " << name() << std::endl;

  //---------- Get forward plate
  ALIPlane plate = getPlate(true, true);
  //---------- Refract while entering splitter
  ALIdouble refra_ind1 = 1.;
  ALIdouble refra_ind2 = findExtraEntryValueMustExist("refra_ind");
  lightray.refract( plate, refra_ind1, refra_ind2 );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Refracted in first plate"); 
  }

  //---------- Get back ward plate (of triangular piiece)
  plate = getPlate(true, false);
  //---------- Refract while exiting prism
  lightray.refract( plate, refra_ind2, refra_ind1 );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Refracted in first plate"); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of deviation of the light ray:
//@@ Reflect in a Plate Splitter
//@@ The beam is reflected in the first plate of the plate splitter, which is obtained without applying the rotation by 'wedge'. 
//@@ After the beam is reflected, it is rotated around the splitter X axis by 'deviX' and around the Y axis by 'deviY'.
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOModifiedRhomboidPrism::fastDeviatesLightRay( LightRay& lightray ) 
{
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST REFLECTION IN MODIFIED RHOMBOID PRISM " << name() << std::endl;

  //---------- Get backward plate
  ALIPlane plate = getPlate(false, false);
  //---------- Intersect with plate
  lightray.intersect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Intersected in plate"); 
  }
  //---------- Deviate Lightray 
  lightray.shiftAndDeviateWhileTraversing( this, 'R');
  /*  ALIdouble deviRX = findExtraEntryValue("deviRX");
  ALIdouble deviRY = findExtraEntryValue("deviRY");
  ALIdouble shiftRX = findExtraEntryValue("shiftRX");
  ALIdouble shiftRY = findExtraEntryValue("shiftRY");
  lightray.shiftAndDeviateWhileTraversing( this, shiftRX, shiftRY, deviRX, deviRY);
  */

  if (ALIUtils::debug >= 2) {
    //    std::cout << " shiftRX " << shiftRX << " shiftRY " << shiftRY << std::endl;
    //   std::cout << " deviRX " << deviRX << " deviRY " << deviRY << std::endl;
    lightray.dumpData("Deviated ");
  }
  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of the light ray traversing
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Traverse Plane Parallel Plate 
//@@ Traslated to the backward plate of the plate splitter
//@@ Shifted in the splitter X direction by 'shiftX', and in the Y direction by 'shiftY' 
//@@ and  rotated around the splitter X axis by 'deviX' and around the Y axis by 'deviY'.
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOModifiedRhomboidPrism::fastTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST TRAVERSE MODIFIED RHOMBOID PRISM " << name() << std::endl;
  
  //---------- Get backward plate
  ALIPlane plate = getPlate(false, false);
  lightray.intersect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Intersected with plate"); 
  }
  //---------- Shift and Deviate
  lightray.shiftAndDeviateWhileTraversing( this, 'T');
  /*  ALIdouble shiftTX = findExtraEntryValue("shiftTX");
  ALIdouble shiftTY = findExtraEntryValue("shiftTY");
  ALIdouble deviTX = findExtraEntryValue("deviTX");
  ALIdouble deviTY = findExtraEntryValue("deviTY");
  lightray.shiftAndDeviateWhileTraversing( this, shiftTX, shiftTY, deviTX, deviTY);*/

  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Shifted and Deviated");
  }

}



//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get one of the rotated plates of an OptO
//@@ 
//@@ The point is defined taking the centre of the prism, 
//@@ and traslating it by +/-1/2 'shift' in the direction of the splitter Z.
//@@ The normal of this plane is obtained as the splitter Z, 
//@@ and then it is rotated around X by 'angle' and with the global rotation matrix. 
//@@ It is also rotated around the splitter X and Y axis by +/-1/2 of the 'wedgeR'. 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIPlane OptOModifiedRhomboidPrism::getRotatedPlate(const ALIbool forwardPlate)
{
  if (ALIUtils::debug >= 4) std::cout << "% LR: GET ROTATED PLATE " << name() << std::endl;
  //---------- Get OptO variables
  const ALIdouble shift = (findExtraEntryValue("shiftRY"));
  ALIdouble wedgeR = findExtraEntryValue("wedgeR");

  //---------- Get centre of plate
  //----- plate centre = OptO centre +/- 1/2 shift
  CLHEP::Hep3Vector plate_point = centreGlob();
  //--- Add to it half of the shift following the direction of the prism Y. -1/2 if it is forward plate, +1/2 if it is backward plate
  ALIdouble normal_sign = -forwardPlate*2 + 1;
  CLHEP::Hep3Vector YAxis(0.,1.,0.);
  CLHEP::HepRotation rmt = rmGlob();
  YAxis = rmt*YAxis;
  plate_point += normal_sign * shift/2. * YAxis;

  //---------- Get normal of plate
  //----- Plate normal before wedgeR (Z axis of OptO rotated 'angle' around X)
  CLHEP::Hep3Vector ZAxis(0.,0.,1.);
  ALIdouble anglePlanes;
  ALIbool we = findExtraEntryValueIfExists("anglePlanes", anglePlanes);
  if( !we ) { 
    anglePlanes = 45.*ALIUtils::deg;
  }
  ZAxis.rotateX( anglePlanes );

  //----- Rotate with global rotation matrix
  CLHEP::Hep3Vector plate_normal = rmt*ZAxis;
  if (ALIUtils::debug >= 3) { 
    ALIUtils::dump3v( plate_point, "plate_point");
    ALIUtils::dump3v( plate_normal, "plate_normal before wedge");
    ALIUtils::dumprm( rmt, "rmt before wedge angles" );
  }

  //----- Rotate plate normal by 1/2 wedgeR angles
  //--- Around X axis
  CLHEP::Hep3Vector XAxis(0.,0.,1.);
  XAxis = rmt*XAxis;
  plate_normal.rotate( normal_sign * wedgeR/2., XAxis );
  if (ALIUtils::debug >= 3) ALIUtils::dump3v( plate_normal, "plate_normal after wedgeR around X ");
  //--- Around the axis obtained rotating the prism Y axis by 'anglePlanes' around the prism X axis
  YAxis = CLHEP::Hep3Vector(0.,1.,0.);
  YAxis.rotateX( anglePlanes );
  YAxis = rmt*YAxis;
  plate_normal.rotate( normal_sign * wedgeR/2., YAxis );
  if (ALIUtils::debug >= 3) ALIUtils::dump3v( plate_normal, "plate_normal after wedgeR around Y ");

  //---------- Return plate plane
  return ALIPlane(plate_point, plate_normal);

}



#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOModifiedRhomboidPrism::fillIguana()
{
  ALIColour* col = new ALIColour( 0., 0., 1., 0. );
  ALIdouble width;
  ALIbool wexists = findExtraEntryValueIfExists("width",width);
  if( !wexists ) width = 1.;
  ALIdouble shift;
  wexists = findExtraEntryValueIfExists("shift",shift);
  if( !wexists ) shift = 4.;
  std::vector<ALIdouble> spar;
  spar.push_back(shift);
  spar.push_back(shift);
  spar.push_back(width);
  spar.push_back(0.);
  spar.push_back(45.);
  spar.push_back(0.);
  IgCocoaFileMgr::getInstance().addSolid( *this, "PARAL", spar, col);
  //add a triangle
  std::vector<ALIdouble> spar2;
  spar2.push_back(width);
  spar2.push_back(width);
  spar2.push_back(0.);
  spar2.push_back(width);
  spar2.push_back(width);
  IgCocoaFileMgr::getInstance().addSolid( *this, "TRD", spar2, col);

}

#endif


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOModifiedRhomboidPrism::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*5.*cm/m, go*5.*cm/m, go*5.*cm/m ); //COCOA internal units are meters
}
