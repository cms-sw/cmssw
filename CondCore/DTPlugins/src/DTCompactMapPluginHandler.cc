/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/10/13 11:54:18 $
 *  $Revision: 1.4 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondCore/DTPlugins/interface/DTCompactMapPluginHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTCompactMapPluginHandler::DTCompactMapPluginHandler() {
//  std::cout << "===================================" << std::endl;
//  std::cout << "=                                 =" << std::endl;
//  std::cout << "=  new DTCompactMapPluginHandler  =" << std::endl;
//  std::cout << "=                                 =" << std::endl;
//  std::cout << "===================================" << std::endl;
//  if ( instance == 0 ) instance = this;
}


//--------------
// Destructor --
//--------------
DTCompactMapPluginHandler::~DTCompactMapPluginHandler() {
}


//--------------
// Operations --
//--------------
void DTCompactMapPluginHandler::build() {
  if ( instance == 0 ) instance = new DTCompactMapPluginHandler;
}


DTReadOutMapping* DTCompactMapPluginHandler::expandMap( const DTReadOutMapping& compMap ) {
  std::vector<DTReadOutGeometryLink> entryList;
  DTReadOutMapping::const_iterator compIter = compMap.begin();
  DTReadOutMapping::const_iterator compIend = compMap.end();
  while ( compIter != compIend ) entryList.push_back( *compIter++ );

  std::string
  rosMap = "expand_";
  rosMap += compMap.mapRobRos();
  std::string
  tdcMap = "expand_";
  tdcMap += compMap.mapCellTdc();
  DTReadOutMapping* fullMap = new DTReadOutMapping( tdcMap, rosMap );
  int ddu;
  int ros;
  int rch;
  int tdc;
  int tch;
  int whe;
  int sta;
  int sec;
  int rob;
  int qua;
  int lay;
  int cel;
  int mt1;
  int mi1;
  int mt2;
  int mi2;
  int def;
  int wha;
  int sea;
  std::vector<DTReadOutGeometryLink>::const_iterator iter = entryList.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend = entryList.end();
  std::vector<DTReadOutGeometryLink>::const_iterator iros = entryList.end();
  std::vector<DTReadOutGeometryLink>::const_iterator irob = entryList.end();
  while ( iter != iend ) {
    const DTReadOutGeometryLink& rosEntry( *iter++ );
    if ( rosEntry.dduId > 0x3fffffff ) continue;
    ddu = rosEntry.dduId;
    ros = rosEntry.rosId;
    whe = rosEntry.wheelId;
    def = rosEntry.stationId;
    sec = rosEntry.sectorId;
    rob = rosEntry.slId;
    mt1 = rosEntry.layerId;
    mi1 = rosEntry.cellId;
    iros = entryList.begin();
    while ( iros != iend ) {
      wha = whe;
      sea = sec;
      const DTReadOutGeometryLink& rchEntry( *iros++ );
      if ( ( rchEntry.dduId != mt1 ) ||
           ( rchEntry.rosId != mi1 ) ) continue;
      rch =  rchEntry.robId;
      if (   rchEntry.wheelId != def   ) wha = rchEntry.wheelId;
      sta =  rchEntry.stationId;
      if (   rchEntry.sectorId != def   ) sea = rchEntry.sectorId;
      rob =  rchEntry.slId;
      mt2 =  rchEntry.layerId;
      mi2 =  rchEntry.cellId;
      irob = entryList.begin();
      while ( irob != iend ) {
        const DTReadOutGeometryLink& robEntry( *irob++ );
        if ( ( robEntry.dduId != mt2 ) ||
             ( robEntry.rosId != mi2 ) ) continue;
        if (   robEntry.robId != rob   ) {
          std::cout << "ROB mismatch " << rob << " "
                                       << robEntry.robId << std::endl;
        }
        tdc =  robEntry.tdcId;
        tch =  robEntry.channelId;
        qua =  robEntry.slId;
        lay =  robEntry.layerId;
        cel =  robEntry.cellId;
        fullMap->insertReadOutGeometryLink( ddu, ros, rch, tdc, tch,
                                                 wha, sta, sea,
                                                 qua, lay, cel );

      }
    }
  }
  return fullMap;
}


