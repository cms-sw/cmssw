//   COCOA class implementation file
//Id:  OptOPlateSplitter.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOPlateSplitter.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/ALIPlane.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
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
void OptOPlateSplitter::detailedDeviatesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED REFLECTION IN PLATE SPLITTER " << name() << std::endl;
  if (ALIUtils::debug >= 3) ALIUtils::dump3v( centreGlob(), " centre Global RF ");

  //---------- Get forward plate
  ALIPlane plate = getPlate(true, true);
  //---------- Reflect
  lightray.reflect( plate );
  if (ALIUtils::debug >= 2) {
    std::cout << "Reflected in plate" << std::endl;
    lightray.dumpData(" "); 
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
void OptOPlateSplitter::detailedTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: DETAILED TRAVERSE IN PLATE SPLITTER " << name() << std::endl;

  //---------- Get forward plate
  ALIPlane plate = getPlate(true, true);
  //---------- If width is 0, just keep the same point 
  ALIdouble width = findExtraEntryValue("width");
  if( width == 0 ) {
    if(ALIUtils::debug >= 3) lightray.dumpData("Traversed with 0 width"); 
    return;
  }

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


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Fast simulation of deviation of the light ray:
//@@ Reflect in a Plate Splitter
//@@ The beam is reflected in the first plate of the plate splitter, which is obtained without applying the rotation by 'wedge'. 
//@@ After the beam is reflected, it is rotated around the splitter X axis by 'deviX' and around the Y axis by 'deviY'.
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOPlateSplitter::fastDeviatesLightRay( LightRay& lightray ) 
{
  if (ALIUtils::debug >= 2) std::cout << "LR: REFLECTION IN PLATE SPLITTER " << name() << std::endl;

  //---------- Get forward plate
  ALIPlane plate = getPlate(true, false);
  //---------- Reflect in plate (including intersection with it)
  lightray.reflect( plate );
  if (ALIUtils::debug >= 2) {
    lightray.dumpData("Reflected in plate"); 
  }
  //---------- Deviate Lightray 
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
void OptOPlateSplitter::fastTraversesLightRay( LightRay& lightray )
{
  if (ALIUtils::debug >= 2) std::cout << "LR: TRAVERSE PLATE SPLITTER  " << name() << std::endl;
  
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
void OptOPlateSplitter::fillIguana()
{
  ALIColour* col = new ALIColour( 0., 0., 0., 0. );
  ALIdouble width;
  ALIbool wexists = findExtraEntryValueIfExists("width",width);
  if( !wexists ) width = 1.;

  std::vector<ALIdouble> spar;
  spar.push_back(4.);
  spar.push_back(4.);
  spar.push_back(width);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, col);
}
#endif


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOPlateSplitter::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*5.*cm/m, go*5.*cm/m, go*1.*cm/m ); //COCOA internal units are meters
}
