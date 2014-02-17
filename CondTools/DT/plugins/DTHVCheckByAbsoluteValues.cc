/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/09/14 13:54:22 $
 *  $Revision: 1.5 $
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
DTHVAbstractCheck::flag DTHVCheckByAbsoluteValues::checkCurrentStatus(
                        int rawId, int type,
                        float valueA, float valueC, float valueS,
                        const std::map<int,timedMeasurement>& snapshotValues,
                        const std::map<int,int>& aliasMap,
                        const std::map<int,int>& layerMap ) {

// find all values for this channel
//  ind dpid = 0;
//  std::map<int,int>::const_iterator lpartIter;
//  std::map<int,int>::const_iterator lpartIend = layerMap.end();
//  if ( ( layerIter = layerMap.find( chp0 ) ) != layerIend ) 
//      dpid = layerIter.second;
//  std::map<int,timedMeasurement>::const_iterator snapIter;
//  std::map<int,timedMeasurement>::const_iterator snapIend =
//                                                 snapshotValues.end();
//  float val1 = -999999.0;
//  float val2 = -999999.0;
//  int chan = dpId * 10;
//  if ( ( snapIter = snapshotValues.find( chan + 1 ) ) != snapIend )
//      val1 = snapIter->second.second;
//  if ( ( snapIter = snapshotValues.find( chan + 2 ) ) != snapIend )
//      val2 = snapIter->second.second;

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

  DTWireId chlId( rawId );
  int part = chlId.wire() - 10;
  DTHVAbstractCheck::flag flag;
  flag.a = flag.c = flag.s = 0;
  if ( type == 1 ) {
    if ( valueA < minHV[part] ) flag.a = 1;
    if ( valueA > maxHV[part] ) flag.a = 2;
    if ( valueS < minHV[   2] ) flag.s = 1;
    if ( valueS > maxHV[   2] ) flag.s = 2;
    if ( valueC < minHV[   3] ) flag.c = 1;
    if ( valueC > maxHV[   3] ) flag.c = 2;
  }
  if ( type == 2 ) {
    float voltA = 0.0;
    float voltS = 0.0;
    float voltC = 0.0;
    DTLayerId lay = chlId.layerId();
    int l_p = chlId.wire();
    DTWireId chA( lay, l_p );
    DTWireId chS( lay, 12 );
    DTWireId chC( lay, 13 );
    std::map<int,int>::const_iterator layerIter;
    std::map<int,int>::const_iterator layerIend = layerMap.end();
    std::map<int,timedMeasurement>::const_iterator snapIter;
    std::map<int,timedMeasurement>::const_iterator snapIend =
                                                   snapshotValues.end();
    int chan;
    if ( ( layerIter = layerMap.find( chA.rawId() ) ) != layerIend ) {
      chan = ( layerIter->second * 10 ) + l_p;
      if ( ( snapIter = snapshotValues.find( chan ) ) != snapIend ) {
        voltA = snapIter->second.second;
      }
    }
    if ( ( layerIter = layerMap.find( chS.rawId() ) ) != layerIend ) {
      chan = ( layerIter->second * 10 ) + 2;
      if ( ( snapIter = snapshotValues.find( chan ) ) != snapIend ) {
        voltS = snapIter->second.second;
      }
    }
    if ( ( layerIter = layerMap.find( chC.rawId() ) ) != layerIend ) {
      chan = ( layerIter->second * 10 ) + 3;
      if ( ( snapIter = snapshotValues.find( chan ) ) != snapIend ) {
        voltC = snapIter->second.second;
      }
    }
    if ( ( valueA > maxCurrent  ) &&
         ( voltA >= minHV[part] ) ) flag.a = 4;
    if ( ( valueS > maxCurrent  ) &&
         ( voltS >= minHV[   2] ) ) flag.s = 4;
    if ( ( valueC > maxCurrent  ) &&
         ( voltC >= minHV[   3] ) ) flag.c = 4;
  }
  return flag;

}


DEFINE_FWK_SERVICE( DTHVCheckByAbsoluteValues );
} }


