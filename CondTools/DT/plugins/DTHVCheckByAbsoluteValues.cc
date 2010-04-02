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
#include "DataFormats/MuonDetId/interface/DTWireId.h"
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
//    minHV[0] = 3500.0;
//    minHV[1] = 3500.0;
//    minHV[2] = 1400.0;
//    minHV[3] =  800.0;
    minHV[0] = 3500.0;
    minHV[1] = 3500.0;
    minHV[2] = 1700.0;
    minHV[3] = 1100.0;
    maxHV[0] = 4000.0;
    maxHV[1] = 4000.0;
    maxHV[2] = 2200.0;
    maxHV[3] = 1600.0;
    maxCurrent = 30.0;
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
              int dpId, int rawId, int type, float value,
              const std::map<int,timedMeasurement>& snapshotValues,
              const std::map<int,int>& aliasMap,
              const std::map<int,int>& layerMap  ) {

  DTWireId chlId( rawId );
  int part = chlId.wire() - 10;
  if ( part < 0 ) return 0;
  if ( part > 3 ) return 0;

// find dp identifier for all channels in this layer
//  DTLayerId lay = chlId.layerId();
//  int chp0 = DTWireId( lay, 10 ).rawId();
//  int chp1 = DTWireId( lay, 11 ).rawId();
//  int chp2 = DTWireId( lay, 12 ).rawId();
//  int chp3 = DTWireId( lay, 13 ).rawId();
//  ind dpi0 = 0;
//  ind dpi1 = 0;
//  ind dpi2 = 0;
//  ind dpi3 = 0;
//  std::map<int,int>::const_iterator layerIter;
//  std::map<int,int>::const_iterator layerIend = layerMap.end();
//  if ( ( layerIter = layerMap.find( chp0 ) ) != layerIend ) 
//      dpi0 = layerIter.second;
//  if ( ( layerIter = layerMap.find( chp1 ) ) != layerIend ) 
//      dpi1 = layerIter.second;
//  if ( ( layerIter = layerMap.find( chp2 ) ) != layerIend ) 
//      dpi2 = layerIter.second;
//  if ( ( layerIter = layerMap.find( chp3 ) ) != layerIend ) 
//      dpi3 = layerIter.second;

// find all values for this channel
  std::map<int,timedMeasurement>::const_iterator snapIter;
  std::map<int,timedMeasurement>::const_iterator snapIend =
                                                 snapshotValues.end();
  float val1 = -999999.0;
  float val2 = -999999.0;
  int chan = dpId * 10;
  if ( ( snapIter = snapshotValues.find( chan + 1 ) ) != snapIend )
      val1 = snapIter->second.second;
  if ( ( snapIter = snapshotValues.find( chan + 2 ) ) != snapIend )
      val2 = snapIter->second.second;

  if ( type == 1 ) {
    if ( value < minHV[part] ) return 1;
    if ( value > maxHV[part] ) return 2;
  }
  if ( type == 2 ) {
    if ( ( value > maxCurrent  ) &&
         ( val1 >= minHV[part] ) ) return 4;
  }
  return 0;

/*
  int status = 0;
  if ( type == 1 ) {
    if ( value < minHV[part] ) status += 1;
    if ( value > maxHV[part] ) status += 2;
  }
  if ( type == 2 ) {
    if ( value > maxCurrent ) status += 4;
  }
  return status;
*/
}


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_SERVICE( DTHVCheckByAbsoluteValues );
} }


