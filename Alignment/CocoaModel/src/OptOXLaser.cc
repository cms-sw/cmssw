//   COCOA class implementation file
//Id:  OptOXLaser.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOXLaser.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#ifdef COCOA_VIS
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#include "Alignment/CocoaVisMgr/interface/ALIColour.h"
#endif
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeTubs.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ default behaviour: create a LightRay object
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOXLaser::defaultBehaviour( LightRay& lightray, Measurement& meas )
{
  if(ALIUtils::debug >= 3) std::cout << "create laser lightray " << std::endl; 
  lightray.startLightRay( this );

}

#ifdef COCOA_VIS
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOXLaser::fillIguana()
{
  //-  std::cout << " filling optosensor " << std::endl;
  //  IgCocoaFileMgr igcomgr = IgCocoaFileMgr::getInstance();
  ALIColour* col = new ALIColour( 1., 0., 0., 0. );
  std::vector<ALIdouble> spar;
  spar.push_back(5.);
  spar.push_back(1.);
  spar.push_back(2.);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, col);
  std::vector<ALIdouble> spar2;
  spar2.push_back(1.);
  spar2.push_back(5.);
  spar2.push_back(2.);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar2, col);
}
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOXLaser::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeTubs( "Tubs", go*0.*cm/m, go*1.*cm/m, go*5.*cm/m ); //COCOA internal units are meters
}

