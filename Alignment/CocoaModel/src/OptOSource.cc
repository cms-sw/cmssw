//   COCOA class implementation file
//Id:  OptOSource.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOSource.h"
#include "Alignment/CocoaModel/interface/LightRay.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include <iostream>
#include <iomanip>
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "CLHEP/Units/SystemOfUnits.h"


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OptOSource::OptOSource()
{ 
  setRmGlobalOriginal( CLHEP::HepRotation() ); 
  setRmGlobal( CLHEP::HepRotation() ); 
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Default behaviour: create a LightRay object
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSource::defaultBehaviour( LightRay& lightray, Measurement& meas ) 
{
  if(ALIUtils::debug >= 3) std::cout << "create laser lightray " << std::endl; 
  lightray.startLightRay( this );
}


#ifdef COCOA_VIS
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
void OptOSource::fillVRML()
{
  ALIVRMLMgr& vrmlmgr = ALIVRMLMgr::getInstance();
  ALIColour* col = new ALIColour( 1., 0., 0., 0. );
  vrmlmgr.AddBox( *this, 0.1, 0.1, 0.2,col);
  vrmlmgr.SendReferenceFrame( *this, 0.12); 
  vrmlmgr.SendName( *this, 0.1 );

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOSource::fillIguana()
{
  ALIColour* col = new ALIColour( 1., 0., 0., 0. );
  std::vector<ALIdouble> spar;
  spar.push_back(1.);
  spar.push_back(2.);
  CLHEP::HepRotation rm;
  rm.rotateX( 90.*deg);
  IgCocoaFileMgr::getInstance().addSolid( *this, "CYLINDER", spar, col, CLHEP::Hep3Vector(), rm);
}
#endif
