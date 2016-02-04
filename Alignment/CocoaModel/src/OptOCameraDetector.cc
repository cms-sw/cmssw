//   COCOA class implementation file
//Id:  OptOCameraDetector.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptOCameraDetector.h"
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>

using namespace CLHEP;

void OptOCameraDetector::participateInMeasurement( LightRay& lightray, Measurement* meas, const ALIstring& behav )
{
 std::cerr << "object not implemented yet " << std::endl;
 exit(1);      
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptOCameraDetector::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*4.*cm/m, go*4.*cm/m, go*1.*cm/m ); //COCOA internal units are meters
}
