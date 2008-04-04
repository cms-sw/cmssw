/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:00:51 $
 *  $Revision: 1.16 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTReadOutMapping::DTReadOutMapping():
  cellMapVersion( " " ),
   robMapVersion( " " ) {
}


DTReadOutMapping::DTReadOutMapping( const std::string& cell_map_version,
                                    const std::string&  rob_map_version ):
  cellMapVersion( cell_map_version ),
   robMapVersion(  rob_map_version ) {

}


DTReadOutGeometryLink::DTReadOutGeometryLink():
     dduId( 0 ),
     rosId( 0 ),
     robId( 0 ),
     tdcId( 0 ),
 channelId( 0 ),
   wheelId( 0 ),
 stationId( 0 ),
  sectorId( 0 ),
      slId( 0 ),
   layerId( 0 ),
    cellId( 0 ) {
}

//--------------
// Destructor --
//--------------
DTReadOutMapping::~DTReadOutMapping() {
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() );
  DTDataBuffer<int,int>::dropBuffer( mapNameGR() );
}

DTReadOutGeometryLink::~DTReadOutGeometryLink() {
}

//--------------
// Operations --
//--------------
int DTReadOutMapping::readOutToGeometry( int      dduId,
                                         int      rosId,
                                         int      robId,
                                         int      tdcId,
                                         int  channelId,
                                         DTWireId& wireId ) const {

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

  int status = readOutToGeometry(      dduId,
                                       rosId,
                                       robId,
                                       tdcId,
                                   channelId,
                                     wheelId,
                                   stationId,
                                    sectorId,
                                        slId,
                                     layerId,
                                      cellId );

  wireId = DTWireId( wheelId, stationId, sectorId, slId, layerId, cellId );
  return status;

}

int DTReadOutMapping::readOutToGeometry( int      dduId,
                                         int      rosId,
                                         int      robId,
                                         int      tdcId,
                                         int  channelId,
                                         int&   wheelId,
                                         int& stationId,
                                         int&  sectorId,
                                         int&      slId,
                                         int&   layerId,
                                         int&    cellId ) const {

  wheelId   =
  stationId =
  sectorId  =
  slId      =
  layerId   =
  cellId    = 0;

  std::string mNameRG = mapNameRG();
  DTBufferTree<int,int>* rgBuf =
  DTDataBuffer<int,int>::findBuffer( mNameRG );
  if ( rgBuf == 0 ) {
    cacheMap();
    rgBuf =
    DTDataBuffer<int,int>::findBuffer( mNameRG );
  }

  std::vector<int> chanKey;
  chanKey.push_back(     dduId );
  chanKey.push_back(     rosId );
  chanKey.push_back(     robId );
  chanKey.push_back(     tdcId );
  chanKey.push_back( channelId );
  int ientry;
  int searchStatus = rgBuf->find( chanKey.begin(), chanKey.end(), ientry );
  if ( !searchStatus ) {
    const DTReadOutGeometryLink& link( readOutChannelDriftTubeMap[ientry] );
      wheelId = link.  wheelId;
    stationId = link.stationId;
     sectorId = link. sectorId;
         slId = link.     slId;
      layerId = link.  layerId;
       cellId = link.   cellId;
  }

  return searchStatus;

}


int DTReadOutMapping::geometryToReadOut( const DTWireId& wireId,
                                         int&     dduId,
                                         int&     rosId,
                                         int&     robId,
                                         int&     tdcId,
                                         int& channelId ) const {
  return geometryToReadOut( wireId.wheel(),
                  wireId.station(),
                  wireId.sector(),
                  wireId.superLayer(),
                  wireId.layer(),
                  wireId.wire(),
                                       dduId,
                                       rosId,
                                       robId,
                                       tdcId,
                                   channelId);
}


int DTReadOutMapping::geometryToReadOut( int    wheelId,
                                         int  stationId,
                                         int   sectorId,
                                         int       slId,
                                         int    layerId,
                                         int     cellId,
                                         int&     dduId,
                                         int&     rosId,
                                         int&     robId,
                                         int&     tdcId,
                                         int& channelId ) const {

  dduId =
  rosId =
  robId =
  tdcId =
  channelId = 0;

  std::string mNameGR = mapNameGR();
  DTBufferTree<int,int>* grBuf =
  DTDataBuffer<int,int>::findBuffer( mNameGR );
  if ( grBuf == 0 ) {
    cacheMap();
    grBuf =
    DTDataBuffer<int,int>::findBuffer( mNameGR );
  }

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  int ientry;
  int searchStatus = grBuf->find( cellKey.begin(), cellKey.end(), ientry );
  if ( !searchStatus ) {
    const DTReadOutGeometryLink& link( readOutChannelDriftTubeMap[ientry] );
        dduId = link.    dduId;
        rosId = link.    rosId;
        robId = link.    robId;
        tdcId = link.    tdcId;
    channelId = link.channelId;
  }

  return searchStatus;

}



const
std::string& DTReadOutMapping::mapCellTdc() const {
  return cellMapVersion;
}


std::string& DTReadOutMapping::mapCellTdc() {
  return cellMapVersion;
}


const
std::string& DTReadOutMapping::mapRobRos() const {
  return robMapVersion;
}


std::string& DTReadOutMapping::mapRobRos() {
  return robMapVersion;
}


void DTReadOutMapping::clear() {
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() );
  DTDataBuffer<int,int>::dropBuffer( mapNameGR() );
  readOutChannelDriftTubeMap.clear();
  return;
}


int DTReadOutMapping::insertReadOutGeometryLink( int     dduId,
                                                 int     rosId,
                                                 int     robId,
                                                 int     tdcId,
                                                 int channelId,
                                                 int   wheelId,
                                                 int stationId,
                                                 int  sectorId,
                                                 int      slId,
                                                 int   layerId,
                                                 int    cellId ) {
  DTReadOutGeometryLink link;
  link.    dduId =     dduId;
  link.    rosId =     rosId;
  link.    robId =     robId;
  link.    tdcId =     tdcId;
  link.channelId = channelId;
  link.  wheelId =   wheelId;
  link.stationId = stationId;
  link. sectorId =  sectorId;
  link.     slId =      slId;
  link.  layerId =   layerId;
  link.   cellId =    cellId;

  int ientry = readOutChannelDriftTubeMap.size();
  readOutChannelDriftTubeMap.push_back( link );

  std::string mNameRG = mapNameRG();
  std::string mNameGR = mapNameGR();

  DTBufferTree<int,int>* rgBuf =
  DTDataBuffer<int,int>::openBuffer( mNameRG );
  DTBufferTree<int,int>* grBuf =
  DTDataBuffer<int,int>::openBuffer( mNameGR );

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  int grStatus =
  grBuf->insert( cellKey.begin(), cellKey.end(), ientry );
  std::vector<int> chanKey;
  chanKey.push_back(     dduId );
  chanKey.push_back(     rosId );
  chanKey.push_back(     robId );
  chanKey.push_back(     tdcId );
  chanKey.push_back( channelId );
  int rgStatus =
  rgBuf->insert( chanKey.begin(), chanKey.end(), ientry );

  if ( grStatus || rgStatus ) return 1;
  else                        return 0;

}


DTReadOutMapping::const_iterator DTReadOutMapping::begin() const {
  return readOutChannelDriftTubeMap.begin();
}


DTReadOutMapping::const_iterator DTReadOutMapping::end() const {
  return readOutChannelDriftTubeMap.end();
}


std::string DTReadOutMapping::mapNameGR() const {
/*
  std::string name = cellMapVersion + "_" + robMapVersion + "_map_GR";
  char nptr[100];
  sprintf( nptr, "%x", reinterpret_cast<unsigned int>( this ) );
  name += nptr;
  return name;
*/
  std::stringstream name;
  name << cellMapVersion << "_" << robMapVersion << "_map_GR" << this;
  return name.str();
}


std::string DTReadOutMapping::mapNameRG() const {
/*
  std::string name = cellMapVersion + "_" + robMapVersion + "_map_RG";
  char nptr[100];
  sprintf( nptr, "%x", reinterpret_cast<unsigned int>( this ) );
  name += nptr;
  return name;
*/
  std::stringstream name;
  name << cellMapVersion << "_" << robMapVersion << "_map_RG" << this;
  return name.str();
}


void DTReadOutMapping::cacheMap() const {

  std::string mNameRG = mapNameRG();
  std::string mNameGR = mapNameGR();

  DTBufferTree<int,int>* rgBuf =
  DTDataBuffer<int,int>::openBuffer( mNameRG );
  DTBufferTree<int,int>* grBuf =
  DTDataBuffer<int,int>::openBuffer( mNameGR );

  int entryNum = 0;
  int entryMax = readOutChannelDriftTubeMap.size();
  while ( entryNum < entryMax ) {

    const DTReadOutGeometryLink& link( readOutChannelDriftTubeMap[entryNum] );

    std::vector<int> cellKey;
    cellKey.push_back( link.  wheelId );
    cellKey.push_back( link.stationId );
    cellKey.push_back( link. sectorId );
    cellKey.push_back( link.     slId );
    cellKey.push_back( link.  layerId );
    cellKey.push_back( link.   cellId );

    grBuf->insert( cellKey.begin(), cellKey.end(), entryNum );

    std::vector<int> chanKey;
    chanKey.push_back( link.    dduId );
    chanKey.push_back( link.    rosId );
    chanKey.push_back( link.    robId );
    chanKey.push_back( link.    tdcId );
    chanKey.push_back( link.channelId );

    rgBuf->insert( chanKey.begin(), chanKey.end(), entryNum );

    entryNum++;

  }

  return;

}

