//   COCOA class implementation file
//Id:  OptOCubeSplitter.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOCubeSplitter.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h" 
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#ifdef COCOA_VIS
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/CocoaVisMgr/interface/ALIColour.h"
#endif
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of deviation of the light ray:
//@@ Reflect in a Cube Splitter
//@@ The beam is reflected in the first plate of the Cube splitter, which is obtained without applying the rotation by 'wedge'. 
//@@ After the beam is reflected, it is rotated around the splitter X axis by 'deviX' and around the Y axis by 'deviY'.
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCubeSplitter::fastDeviatesLightRay( LightRay& lightray ) 
{
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST REFLECTION IN CUBE SPLITTER " << name() << std::endl;

  //---------- Get forward plate
  ALIPlane plate = getMiddlePlate();
  //---------- Reflect in plate (including intersection with it)
  lightray.reflect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Reflected in plate"); 
  }
  //---------- Deviate Lightray 
  //  ALIdouble deviRX = findExtraEntryValue("deviRX");
  // ALIdouble deviRY = findExtraEntryValue("deviRY");
  //  lightray.shiftAndDeviateWhileTraversing( this, 0., 0., 0., deviRX, deviRY, 0.);
  lightray.shiftAndDeviateWhileTraversing( this, 'R' );
  if (ALIUtils::debug >= 2) {
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
void OptOCubeSplitter::fastTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST TRAVERSE CUBE SPLITTER  " << name() << std::endl;
  
  //---------- Get backward plate
  ALIPlane plate = getPlate(false, false);
  lightray.intersect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Intersected with plate"); 
  }
  //---------- Shift and Deviate
  lightray.shiftAndDeviateWhileTraversing( this, 'T');
  /*  ALIdouble shiftX = findExtraEntryValue("shiftX");
  ALIdouble shiftY = findExtraEntryValue("shiftY");
  ALIdouble deviTX = findExtraEntryValue("deviTX");
  ALIdouble deviTY = findExtraEntryValue("deviTY");
  lightray.shiftAndDeviateWhileTraversing( this, shiftX, shiftY, deviTX, deviTY);*/

  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Shifted and Deviated");
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of deviation of the light ray (reflection, shift, ...)

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of Reflection in Cube Splitter
//@@ The software gets the plane of entering as the forward splitter plane and the beam is refracted
//@@ The software gets the plane of reflection as the middle splitter plane
//@@ The beam is reflected in this plane.
//@@ The beam is refracted at exiting 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCubeSplitter::detailedDeviatesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED REFLECTION IN CUBE SPLITTER " << name() << std::endl;

  if(ALIUtils::debug >= 2) ALIUtils::dump3v( centreGlob(), "centreGlob");
  //---------- Get first plate
  if (ALIUtils::debug >= 3) std::cout << "%%%%% refracting at entering first plate " << std::endl; 
  if (ALIUtils::debug >= 3) std::cout << "%%% getting first plate " << std::endl; 
  ALIPlane plate = getPlate(true, true);

  //---------- Refract  
  ALIdouble refra_ind1 = 1.;
  ALIdouble refra_ind2 = findExtraEntryValue("refra_ind");
  lightray.refract( plate, refra_ind1, refra_ind2 );

  //---------- Get middle plate
  if (ALIUtils::debug >= 3) std::cout << "%%%%% reflecting in middle plate " << std::endl; 
  if (ALIUtils::debug >= 3) std::cout << "%%% getting middle plate " << std::endl; 
  plate = getMiddlePlate();

  //---------- Reflect
  lightray.reflect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Reflected in plate"); 
  }

  //--------- Get upper plate
  if (ALIUtils::debug >= 3) std::cout << "%%%%% getting second plate " << std::endl; 
  plate = getUpperPlate();
  if (ALIUtils::debug >= 3) std::cout << "%%%%% refracting at exiting second plate " << std::endl; 
  lightray.refract( plate, refra_ind2, refra_ind1 );

 if (ALIUtils::debug >= 2) {
    lightray.dumpData("After CubeSplitter"); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of the light ray traversing
//@@  The beam enters the splitter, is refracted, traverses the splitter and finally is again refracted when it exits:
//@@ Get the intersection with the forward splitter plane
//@@ Refract the beam and propagate until it intersects the backward splitter plane.
//@@ Finally the beam is refracted again.
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCubeSplitter::detailedTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED TRAVERSE CUBE SPLITTER " << name() << std::endl;

  //---------- Get forward plate
  ALIPlane plate = getPlate(true, true);

  //---------- Refract while entering splitter
  ALIdouble refra_ind1 = 1.;
  ALIdouble refra_ind2 = findExtraEntryValue("refra_ind");
  lightray.refract( plate, refra_ind1, refra_ind2 );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Refracted in first plate"); 
  }

  //---------- Get backward plate
  plate = getPlate(false, true);
  //---------- Refract while exiting splitter
  lightray.refract( plate, refra_ind2, refra_ind1 );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Refracted in first plate"); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get the middle semitransparent plate of a Cube Splittter
//@@ The point is defined taking the centre of the splitter, 
//@@ The normal of this plane is obtained as the splitter Z rotated 45o around X
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIPlane OptOCubeSplitter::getMiddlePlate()
{
  if (ALIUtils::debug >= 4) std::cout << "%%% LR: GET MIDDLE PLATE " << name() << std::endl;
  //---------- Get centre and normal of plate
  //----- plate normal before wedge (Z axis of OptO)
  ALIdouble anglePlanes;
  ALIbool angpl = findExtraEntryValueIfExists("anglePlanes", anglePlanes);
  if( !angpl ) {
    anglePlanes = acos(0.)/2.;  //default is 45o  !!! this creates problem in 'isr_medidas_globales.txt': laser goes along X and does not intersect cube if angles Y 0, anglePlanes 45 
    if (ALIUtils::debug >= 4) std::cout << "anglePlanes default = " << anglePlanes/deg << std::endl;
  }
  CLHEP::Hep3Vector Axis(0., 0., 1.);
  CLHEP::Hep3Vector XAxis(1., 0., 0.);
  Axis.rotate( anglePlanes, XAxis);
  CLHEP::HepRotation rmt = rmGlob();
  CLHEP::Hep3Vector plate_normal = rmt*Axis;
  if (ALIUtils::debug >= 3) { 
    ALIUtils::dumprm( rmt, "rmt before wedge angles" );
    ALIUtils::dump3v( plate_normal, "plate_normal before wedge");
  }
  //----- plate centre = OptO 
  CLHEP::Hep3Vector plate_point = centreGlob();

  //---------- Return plate plane
  return ALIPlane(plate_point, plate_normal);

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get the upper plate of a Cube Splittter
//@@ The point is defined taking the centre of the splitter, 
//@@ The normal of this plane is obtained as the splitter Z rotated 45o around X
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIPlane OptOCubeSplitter::getUpperPlate()
{
  if (ALIUtils::debug >= 4) std::cout << "LR: GET UPPER PLATE " << name() << std::endl;
  //---------- Get centre and normal of plate
  ALIdouble width = findExtraEntryValue("width");
  //----- plate normal before wedge (Y axis of OptO)
  CLHEP::Hep3Vector Axis(0., 1., 0.);
  CLHEP::HepRotation rmt = rmGlob();
  CLHEP::Hep3Vector plate_normal = rmt*Axis;
  if (ALIUtils::debug >= 3) { 
    ALIUtils::dumprm( rmt, "rmt before wedge angles" );
    ALIUtils::dump3v( plate_normal, "plate_normal before wedge");
  }
  //----- plate centre = OptO centre +1/2 width in Y direction 
  CLHEP::Hep3Vector plate_point = centreGlob();
  plate_point += width/2. * plate_normal;

  //---------- Return plate plane
  return ALIPlane(plate_point, plate_normal);

}


#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCubeSplitter::fillIguana()
{
  ALIColour* col = new ALIColour( 0., 0., 0., 0. );
  ALIdouble width;
  ALIbool wexists = findExtraEntryValueIfExists("width",width);
  if( !wexists ) width = 4.;

  std::vector<ALIdouble> spar;
  spar.push_back(width);
  spar.push_back(width);
  spar.push_back(width);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, col);
}
#endif


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCubeSplitter::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*5.*cm/m, go*5.*cm/m, go*5.*cm/m ); //COCOA internal units are meters
}
