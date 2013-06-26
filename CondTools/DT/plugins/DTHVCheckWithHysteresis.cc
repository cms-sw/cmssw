/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/09/14 14:30:26 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/plugins/DTHVCheckWithHysteresis.h"

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
DTHVCheckWithHysteresis::DTHVCheckWithHysteresis(
                           const edm::ParameterSet & iConfig, 
                           edm::ActivityRegistry & iAR ) {
  if ( instance == 0 ) {
    std::cout << "create DTHVCheckWithHysteresis" << std::endl;
    minHVl = new float[4];
    minHVh = new float[4];
    maxHV = new float[4];
    minHVl[0] = 3000.0;
    minHVl[1] = 3000.0;
    minHVl[2] = 1200.0;
    minHVl[3] =  600.0;
    minHVh[0] = 3500.0;
    minHVh[1] = 3500.0;
    minHVh[2] = 1700.0;
    minHVh[3] = 1100.0;
    maxHV[0] = 4000.0;
    maxHV[1] = 4000.0;
    maxHV[2] = 2200.0;
    maxHV[3] = 1600.0;
    maxCurrent = 30.0;
    oldStatusA = new std::map<int,int>;
    oldStatusC = new std::map<int,int>;
    oldStatusS = new std::map<int,int>;
    instance = this;
  }
}

//--------------
// Destructor --
//--------------
DTHVCheckWithHysteresis::~DTHVCheckWithHysteresis() {
}

//--------------
// Operations --
//--------------
//int DTHVCheckWithHysteresis::checkCurrentStatus(
DTHVAbstractCheck::flag DTHVCheckWithHysteresis::checkCurrentStatus(
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

  float minHV[4];
//  DTLayerId lay = chlId.layerId();
//  int chp0 = DTWireId( lay, 10 ).rawId();
//  int chp1 = DTWireId( lay, 11 ).rawId();
//  int chp2 = DTWireId( lay, 12 ).rawId();
//  int chp3 = DTWireId( lay, 13 ).rawId();

  DTWireId chlId( rawId );
  int part = chlId.wire() - 10;
  DTHVAbstractCheck::flag flag;
  flag.a = flag.c = flag.s = 0;

  std::map<int,int>::iterator chanIter;
  if ( ( ( chanIter = oldStatusA->find( rawId ) ) != oldStatusA->end() ) &&
       (   chanIter->second % 2 ) ) minHV[part] = minHVh[part];
  else                              minHV[part] = minHVl[part];
  if ( ( ( chanIter = oldStatusS->find( rawId ) ) != oldStatusS->end() ) &&
       (   chanIter->second % 2 ) ) minHV[   2] = minHVh[   2];
  else                              minHV[   2] = minHVl[   2];
  if ( ( ( chanIter = oldStatusC->find( rawId ) ) != oldStatusC->end() ) &&
       (   chanIter->second % 2 ) ) minHV[   3] = minHVh[   3];
  else                              minHV[   3] = minHVl[   3];

  if ( type == 1 ) {
    if ( valueA < minHV[part] ) flag.a = 1;
    if ( valueA > maxHV[part] ) flag.a = 2;
    if ( valueS < minHV[   2] ) flag.s = 1;
    if ( valueS > maxHV[   2] ) flag.s = 2;
    if ( valueC < minHV[   3] ) flag.c = 1;
    if ( valueC > maxHV[   3] ) flag.c = 2;
    if ( ( chanIter = oldStatusA->find( rawId ) ) == oldStatusA->end() )
         oldStatusA->insert( std::pair<int,int>( rawId, flag.a ) );
    else chanIter->second = flag.a;
    if ( ( chanIter = oldStatusC->find( rawId ) ) == oldStatusC->end() )
         oldStatusC->insert( std::pair<int,int>( rawId, flag.c ) );
    else chanIter->second = flag.c;
    if ( ( chanIter = oldStatusS->find( rawId ) ) == oldStatusS->end() )
         oldStatusS->insert( std::pair<int,int>( rawId, flag.s ) );
    else chanIter->second = flag.s;
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


void DTHVCheckWithHysteresis::setStatus(
                        int rawId,
                        int flagA, int flagC, int flagS,
                        const std::map<int,timedMeasurement>& snapshotValues,
                        const std::map<int,int>& aliasMap,
                        const std::map<int,int>& layerMap ) {
//  std::cout << "set status " << rawId << " "
//            << flagA << " " << flagC << " " << flagS << std::endl;
  std::map<int,int>::iterator chanIter;
  if ( ( chanIter = oldStatusA->find( rawId ) ) == oldStatusA->end() )
       oldStatusA->insert( std::pair<int,int>( rawId, flagA ) );
  else chanIter->second = flagA;
  if ( ( chanIter = oldStatusC->find( rawId ) ) == oldStatusA->end() )
       oldStatusC->insert( std::pair<int,int>( rawId, flagC ) );
  else chanIter->second = flagC;
  if ( ( chanIter = oldStatusS->find( rawId ) ) == oldStatusA->end() )
       oldStatusS->insert( std::pair<int,int>( rawId, flagS ) );
  else chanIter->second = flagS;
  return;
}


DEFINE_FWK_SERVICE( DTHVCheckWithHysteresis );
} }


