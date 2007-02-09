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
//
//
//--------------------------------------------------

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
DTSCTrigUnit::DTSCTrigUnit(DTChamber* stat, edm::ParameterSet& tu_pset) {

  bool geom_debug = tu_pset.getUntrackedParameter<bool>("Debug");
  edm::ParameterSet bti_conf     = tu_pset.getParameter<edm::ParameterSet>("BtiParameters");
  edm::ParameterSet traco_conf   = tu_pset.getParameter<edm::ParameterSet>("TracoParameters");
  edm::ParameterSet tstheta_conf = tu_pset.getParameter<edm::ParameterSet>("TSThetaParameters");
  edm::ParameterSet tsphi_conf   = tu_pset.getParameter<edm::ParameterSet>("TSPhiParameters");

  // create the geometry from the station
  _geom = new DTTrigGeom(stat, geom_debug);

  // create BTI
  _theBTIs = new DTBtiCard(_geom, bti_conf);

  // create TSTheta
  _theTSTheta = new DTTSTheta(_geom,_theBTIs,tstheta_conf);

  // create TRACO
  _theTRACOs = new DTTracoCard(_geom,_theBTIs,_theTSTheta, traco_conf);

  // create TSPhi
  _theTSPhi = new DTTSPhi(_geom,_theTRACOs,tsphi_conf);


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






