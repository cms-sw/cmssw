/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/01/18 18:36:32 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTHVCheckByAbsoluteValues.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

namespace cond { namespace service {

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTHVCheckByAbsoluteValues::DTHVCheckByAbsoluteValues(
                           const edm::ParameterSet & iConfig, 
                           edm::ActivityRegistry & iAR ) {
  if ( instance == 0 ) {
    std::cout << "create DTHVCheckByAbsoluteValues" << std::endl;
    minHV = new float[4];
    maxHV = new float[4];
    minHV[0] = 3500.0;
    minHV[1] = 3500.0;
    minHV[2] = 1700.0;
    minHV[3] = 1100.0;
    maxHV[0] = 4000.0;
    maxHV[1] = 4000.0;
    maxHV[2] = 2200.0;
    maxHV[3] = 1600.0;
    maxCurrent = 3.0;
    instance = this;
  }
}

//--------------
// Destructor --
//--------------
DTHVCheckByAbsoluteValues::~DTHVCheckByAbsoluteValues() {
}

//--------------
// Operations --
//--------------
int DTHVCheckByAbsoluteValues::checkCurrentStatus(
                               int part, int type, float value ) {
  if ( part < 0 ) return 0;
  if ( part > 3 ) return 0;
  int status = 0;
  if ( type == 1 ) {
    if ( value < minHV[part] ) status += 1;
    if ( value > maxHV[part] ) status += 2;
  }
  if ( type == 2 ) {
    float maxCurrent = 10.0;
    if ( value > maxCurrent ) status += 4;
  }
  return status;
}



DEFINE_FWK_SERVICE( DTHVCheckByAbsoluteValues );
} }


