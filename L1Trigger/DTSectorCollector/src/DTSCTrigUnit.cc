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
//   09/01/07 C. Battilana : moved to local conf
//   30/03/07 SV : configuration through DTConfigManager
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/DTSectorCollector/interface/DTSCTrigUnit.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"


//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <cmath>

//----------------
// Constructors --
//----------------
DTSCTrigUnit::DTSCTrigUnit(DTChamber *stat, DTTTrigBaseSync *sync) {

  DTChamberId chambid = stat->id();
  // bool geom_debug = conf_manager->getDTConfigTrigUnit(chambid)->debug();

  // create the geometry from the station
  //_geom = new DTTrigGeom(stat, geom_debug);
  _geom = new DTTrigGeom(stat, false); // CB FIXME: update when debug will be read via PSet

  // create BTI
  _theBTIs = new DTBtiCard(_geom, sync);

  // create TSTheta
  _theTSTheta = new DTTSTheta(_geom, _theBTIs);

  // create TRACO
  _theTRACOs = new DTTracoCard(_geom, _theBTIs, _theTSTheta);

  // create TSPhi
  _theTSPhi = new DTTSPhi(_geom, _theTRACOs);

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

void DTSCTrigUnit::setConfig(const DTConfigManager *conf){

  _theBTIs->setConfig(conf);
  _theTSTheta->setConfig(conf);
  _theTRACOs->setConfig(conf);
  _theTSPhi->setConfig(conf);

}  





