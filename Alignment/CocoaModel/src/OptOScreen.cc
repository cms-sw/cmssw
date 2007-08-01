//   COCOA class implementation file
//Id:  OptOScreen.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOScreen.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include <iostream>
#include <iomanip>
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ do nothing
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOScreen::defaultBehaviour( LightRay& lightray, Measurement& meas ) 
{
 
}
#ifdef COCOA_VIS

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOScreen::fillVRML()
{
  //-  std::cout << " filling optosensor " << std::endl;
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  ALIColour* col = new ALIColour( 1., 0., 0.5, 0. );
  vrmlmgr.AddBox( *this, .6, .6, .1, col);
  vrmlmgr.SendReferenceFrame( *this, 0.6); 
  vrmlmgr.SendName( *this, 0.01 );
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOScreen::fillIguana()
{
  ALIColour* col = new ALIColour( 0., 1., 1., 0. );
  std::vector<ALIdouble> spar;
  spar.push_back(6.);
  spar.push_back(6.);
  spar.push_back(1.);
  IgCocoaFileMgr::getInstance().addSolid( *this, "BOX", spar, col);
}
#endif
