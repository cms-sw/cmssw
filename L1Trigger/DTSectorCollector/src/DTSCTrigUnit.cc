//-------------------------------------------------
//
//   Class: DTSCTrigUnit.cpp
//
//   Description: Muon Barrel Trigger Unit (Chamber trigger)
//
//
//   Author List:
//   C. Grandi
//   Modifications:
//
//
//--------------------------------------------------

//#include "Utilities/Configuration/interface/Architecture.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSCTrigUnit.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cmath>

//----------------
// Constructors --
//----------------
DTSCTrigUnit::DTSCTrigUnit(DTChamber* stat, DTConfig* conf) : 
                                                          _config(conf) {


  // create the geometry from the station
  _geom = new DTTrigGeom(stat, _config);

  // create BTI
  _theBTIs = new DTBtiCard(_geom);

  // create TSTheta
  _theTSTheta = new DTTSTheta(_geom,_theBTIs);

  // create TRACO
  _theTRACOs = new DTTracoCard(_geom,_theBTIs,_theTSTheta);

  // create TSPhi
  _theTSPhi = new DTTSPhi(_geom,_theTRACOs);


}


//--------------
// Destructor --
//--------------
DTSCTrigUnit::~DTSCTrigUnit(){
  delete _theBTIs;
  delete _theTRACOs;
  delete _theTSPhi;
  delete _theTSTheta;
  delete _geom;
}

DTGeomSupplier* 
DTSCTrigUnit::GeomSupplier(const DTTrigData* trig) const {
    //@@ patch for Sun 4.2 compiler
    DTTrigData* tr = const_cast<DTTrigData*>(trig);
    if(dynamic_cast<DTBtiTrigData*>(tr))return _theBTIs;
    if(dynamic_cast<DTTracoTrigData*>(tr))return _theTRACOs;
    if(dynamic_cast<DTChambPhSegm*>(tr))return _theTSPhi;
    if(dynamic_cast<DTChambThSegm*>(tr))return _theTSTheta;
    //    if(dynamic_cast<const DTBtiTrigData*>(trig))return _theBTIs;
    //    if(dynamic_cast<const DTTracoTrigData*>(trig))return _theTRACOs;
    //    if(dynamic_cast<const DTChambPhSegm*>(trig))return _theTSPhi;
    //    if(dynamic_cast<const DTChambThSegm*>(trig))return _theTSTheta;
    return 0;
  }






