//   COCOA class implementation file
//Id:  OptORisleyPrism.cc
//CAT: Model
//
//   History: v1.0 
//   Pedro Arce

#include "Alignment/CocoaModel/interface/OptORisleyPrism.h"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "Alignment/CocoaDDLObjects/interface/CocoaSolidShapeBox.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"

using namespace CLHEP;

void OptORisleyPrism::participateInMeasurement( LightRay& lightray, Measurement& meas, const ALIstring& behav )
{
 std::cerr << "object not implemented yet " << std::endl;
 exit(1);      
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void OptORisleyPrism::constructSolidShape()
{
  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("VisScale", go );

  theSolidShape = new CocoaSolidShapeBox( "Box", go*5.*cm/m, go*5.*cm/m, go*5.*cm/m ); //COCOA internal units are meters
}
