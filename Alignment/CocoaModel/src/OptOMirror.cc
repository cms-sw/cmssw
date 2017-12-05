//   COCOA class implementation file
//Id:  OptOMirror.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOMirror.h"
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

  //---------- Default behaviour: create a LightRay object
void OptOMirror::defaultBehaviour( LightRay& lightray, Measurement& meas ) 
{
  detailedDeviatesLightRay( lightray );
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Detailed simulation of Reflection in Mirror
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOMirror::detailedDeviatesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED REFLECTION IN MIRROR " << name() << std::endl;
  if (ALIUtils::debug >= 3) ALIUtils::dump3v( centreGlob(), " centre Global ");

  //---------- Get forward plate and intersect lightray with it
  ALIPlane plate = getPlate(true, false);
  lightray.intersect( plate );
  CLHEP::Hep3Vector inters = lightray.point( );

  //---------- Get centre of forward plate
  //----- Get Z axis
  CLHEP::Hep3Vector ZAxis = getZAxis();
  CLHEP::HepRotation rmt = rmGlob();
  ZAxis = rmt * ZAxis;
  //----- Get centre
  ALIdouble width = findExtraEntryValue("width");
  CLHEP::Hep3Vector plate_centre = centreGlob() - 0.5 * width * ZAxis; 
  //-  if(ALIUtils::debug >= 4) std::cout << " mirror width " << width << std::endl;

  //---------- Get the distance between the intersection point and the centre of the forward plate
  ALIdouble distance = ( plate_centre - inters ).mag();

  //---------- Get normal to mirror at intersection point 
  //------- Get angle of mirror surface (angle between plate centre and intersection)
  ALIdouble flatness = findExtraEntryValue("flatness");
  //-- flatness is defined as number of 632 nm wavelengths
  flatness *= 632.E-9;
  ALIdouble length = 0.;

  ALIdouble curvature_radius;
  ALIdouble angFlatness;
  if( flatness != 0) {
    length = findExtraEntryValueMustExist("length");
    curvature_radius = (flatness*flatness + length*length) / (2*flatness);
    angFlatness = asin( distance / curvature_radius);
  } else {
    curvature_radius = ALI_DBL_MAX;
    angFlatness = 0;
  }

  if (ALIUtils::debug >= 3) {
    std::cout << " intersection with plate " << inters << std::endl;
    std::cout << " plate_centre " << plate_centre << std::endl;
    std::cout << " distance plate_centre - intersection " << distance << std::endl;
    std::cout << " flatness " << flatness << ", length " << length;
    std::cout << ", curvature radius " << curvature_radius << " angle of flatness " << angFlatness << std::endl;
  }

  //----- Axis of rotation is perpendicular to Z Axis and to line plate_centre - intersection
  CLHEP::Hep3Vector ipcV = inters - plate_centre;
  if( ipcV.mag() != 0) ipcV *= 1./ipcV.mag();
  CLHEP::HepRotation rtm = rmGlob();
  ipcV = rtm*ipcV;
  CLHEP::Hep3Vector rotationAxis = ipcV.cross(ZAxis);
  //----- normal is object normal rotated around this axis
  CLHEP::Hep3Vector inters_normal = CLHEP::Hep3Vector(0.,0.,1.);
  inters_normal.rotate( angFlatness, rotationAxis );  
  inters_normal = rmt * inters_normal;

  if (ALIUtils::debug >= 2) {
     ALIUtils::dump3v( ipcV, " intersection -  plate_centre std::vector ");
     std::cout << "rotation Axis " << rotationAxis << std::endl;
     std::cout << " plate normal at intersection point " << inters_normal << std::endl;
}
  //---------- Reflect in plate 
  ALIdouble cosang = -( inters_normal * lightray.direction() ) / 
           inters_normal.mag() / lightray.direction().mag();
  CLHEP::Hep3Vector lrold = lightray.direction();
  lightray.setDirection( lightray.direction() + inters_normal*2*cosang );

  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Reflected in mirror"); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of Reflection in Mirror
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOMirror::fastDeviatesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: FAST REFLECTION IN MIRROR " << name() << std::endl;

  //---------- Get forward plate
  ALIPlane plate = getPlate(true, false);

  //---------- Reflect in plate (including intersection with it)
  lightray.reflect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Reflected in plate"); 
  }
  //---------- Deviate Lightray 
  //  ALIdouble deviX = findExtraEntryValue("deviX");
  // ALIdouble deviY = findExtraEntryValue("deviY");
  //  lightray.shiftAndDeviateWhileTraversing( this, 0., 0., 0., deviX, deviY, 0.);
  lightray.shiftAndDeviateWhileTraversing( this, 'R' );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Deviated ");
  }

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOMirror::detailedTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED TRAVERSE IN MIRROR " << name() << std::endl;

  //---------- Get forward plate
  ALIPlane plate = getPlate(true, true);
  //---------- If width is 0, just keep the same point 
  ALIdouble width = findExtraEntryValue("width");
  if( width == 0 ) {
    if(ALIUtils::debug >= 3) lightray.dumpData("Traversed with 0 width"); 
    return;
  }

  //---------- Refract while entering mirror
  ALIdouble refra_ind1 = 1.;
  ALIdouble refra_ind2 = findExtraEntryValue("refra_ind");
  lightray.refract( plate, refra_ind1, refra_ind2 );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Refracted in first plate"); 
  }

  //---------- Get backward plate
  plate = getPlate(false, true);
  //---------- Refract while exiting mirror
  lightray.refract( plate, refra_ind2, refra_ind1 );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Refracted in first plate"); 
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOMirror::fastTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: TRAVERSE MIRROR  " << name() << std::endl;
  
  //---------- Get backward plate
  ALIPlane plate = getPlate(false, false);
  lightray.intersect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Intersected with plate"); 
  }
  //---------- Shift and Deviate
  lightray.shiftAndDeviateWhileTraversing( this, 'T' );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Shifted and Deviated");
  }

}



#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOMirror::fillIguana()
{
  ALIdouble width;
  ALIbool wexists = findExtraEntryValueIfExists("width",width);
  if( !wexists ) width = 1.;
  ALIdouble length;
  wexists = findExtraEntryValueIfExists("length",length);
  if( !wexists ) length = 4.;

  ALIColour* col = new ALIColour( 0., 0., 1., 0. );
  std::vector<ALIdouble> spar;
  spar.push_back(length);
  spar.push_back(length);
  spar.push_back(width);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, col);
}
#endif


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOMirror::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*5.*cm/m, go*5.*cm/m, go*1.*cm/m ); //COCOA internal units are meters
}
