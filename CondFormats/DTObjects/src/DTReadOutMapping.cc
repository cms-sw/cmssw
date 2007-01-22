/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/07/19 09:32:10 $
 *  $Revision: 1.12 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTDataBuffer.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------


//---------------
// C++ Headers --
//---------------


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
  std::string mapRtoG =
       cellMapVersion + "_" + robMapVersion + "_map_RG";
  std::string mapGtoR =
       cellMapVersion + "_" + robMapVersion + "_map_GR";
  DTDataBuffer<int,int>::dropBuffer( mapRtoG );
  DTDataBuffer<int,int>::dropBuffer( mapGtoR );
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

  std::string mapRtoG =
       cellMapVersion + "_" + robMapVersion + "_map_RG";
  DTBufferTree<int,int>* rgBuf =
                         DTDataBuffer<int,int>::findBuffer( mapRtoG );
  if ( rgBuf == 0 ) {
    initSetup();
    rgBuf = DTDataBuffer<int,int>::findBuffer( mapRtoG );
  }
  std::vector<int> chanKey;
  chanKey.push_back(     dduId );
  chanKey.push_back(     rosId );
  chanKey.push_back(     robId );
  chanKey.push_back(     tdcId );
  chanKey.push_back( channelId );
  int geometryId = 0;
  int searchStatus = rgBuf->find( chanKey.begin(), chanKey.end(), geometryId );
  if ( !searchStatus ) {
       cellId = geometryId %  100;
                geometryId /= 100;
      layerId = geometryId %  10;
                geometryId /= 10;
         slId = geometryId %  10;
                geometryId /= 10;
     sectorId = geometryId %  100;
                geometryId /= 100;
    stationId = geometryId %  10;
                geometryId /= 10;
      wheelId = geometryId %  10;
      wheelId -= 5;
  }

  return searchStatus;

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

  std::string mapGtoR =
       cellMapVersion + "_" + robMapVersion + "_map_GR";
  DTBufferTree<int,int>* grBuf = DTDataBuffer<int,int>::findBuffer( mapGtoR );
  if ( grBuf == 0 ) {
    initSetup();
    grBuf = DTDataBuffer<int,int>::findBuffer( mapGtoR );
  }
  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  int readoutId = 0;
  int searchStatus = grBuf->find( cellKey.begin(), cellKey.end(), readoutId );
  if ( !searchStatus ) {
    channelId = readoutId %  100;
                readoutId /= 100;
        tdcId = readoutId %  10;
                readoutId /= 10;
        robId = readoutId %  10;
                readoutId /= 10;
        rosId = readoutId %  100;
                readoutId /= 100;
        dduId = readoutId %  1000;
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

  readOutChannelDriftTubeMap.push_back( link );

  std::string mapRtoG =
       cellMapVersion + "_" + robMapVersion + "_map_RG";
  std::string mapGtoR =
       cellMapVersion + "_" + robMapVersion + "_map_GR";

  DTBufferTree<int,int>* rgBuf =
                         DTDataBuffer<int,int>::openBuffer( mapRtoG );
  DTBufferTree<int,int>* grBuf =
                         DTDataBuffer<int,int>::openBuffer( mapGtoR );

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  int readoutId = (     dduId *  1000000 ) +
                  (     rosId *    10000 ) +
                  (     robId *     1000 ) +
                  (     tdcId *      100 ) +
                    channelId;
  int grStatus =
  grBuf->insert( cellKey.begin(), cellKey.end(), readoutId );
  std::vector<int> chanKey;
  chanKey.push_back(     dduId );
  chanKey.push_back(     rosId );
  chanKey.push_back(     robId );
  chanKey.push_back(     tdcId );
  chanKey.push_back( channelId );
  int geometryId = (   wheelId *  10000000 ) +
                   ( stationId *   1000000 ) +
                   (  sectorId *     10000 ) +
                   (      slId *      1000 ) +
                   (   layerId *       100 ) +
                        cellId;
  int rgStatus =
  rgBuf->insert( chanKey.begin(), chanKey.end(), geometryId );

  if ( grStatus || rgStatus ) return 1;
  else                        return 0;

}


DTReadOutMapping::const_iterator DTReadOutMapping::begin() const {
  return readOutChannelDriftTubeMap.begin();
}


DTReadOutMapping::const_iterator DTReadOutMapping::end() const {
  return readOutChannelDriftTubeMap.end();
}


void DTReadOutMapping::initSetup() const {

  std::string mapRtoG =
       cellMapVersion + "_" + robMapVersion + "_map_RG";
  std::string mapGtoR =
       cellMapVersion + "_" + robMapVersion + "_map_GR";

  DTBufferTree<int,int>* rgBuf =
                         DTDataBuffer<int,int>::openBuffer( mapRtoG );
  DTBufferTree<int,int>* grBuf =
                         DTDataBuffer<int,int>::openBuffer( mapGtoR );

  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
  int      dduId;
  int      rosId;
  int      robId;
  int      tdcId;
  int  channelId;
  int  readoutId;
  int    wheelId;
  int  stationId;
  int   sectorId;
  int       slId;
  int    layerId;
  int     cellId;
  int geometryId;
  while ( iter != iend ) {

    const DTReadOutGeometryLink& link = *iter++;
        dduId = link.    dduId;
        rosId = link.    rosId;
        robId = link.    robId;
        tdcId = link.    tdcId;
    channelId = link.channelId;
      wheelId = link.  wheelId + 5;
    stationId = link.stationId;
     sectorId = link. sectorId;
         slId = link.     slId;
      layerId = link.  layerId;
       cellId = link.   cellId;
   
    std::vector<int> cellKey;
    cellKey.push_back(   wheelId );
    cellKey.push_back( stationId );
    cellKey.push_back(  sectorId );
    cellKey.push_back(      slId );
    cellKey.push_back(   layerId );
    cellKey.push_back(    cellId );
    readoutId = (     dduId *  1000000 ) +
                (     rosId *    10000 ) +
                (     robId *     1000 ) +
                (     tdcId *      100 ) +
                  channelId;
    grBuf->insert( cellKey.begin(), cellKey.end(), readoutId );

    std::vector<int> chanKey;
    chanKey.push_back(     dduId );
    chanKey.push_back(     rosId );
    chanKey.push_back(     robId );
    chanKey.push_back(     tdcId );
    chanKey.push_back( channelId );
    geometryId = (   wheelId *  10000000 ) +
                 ( stationId *   1000000 ) +
                 (  sectorId *     10000 ) +
                 (      slId *      1000 ) +
                 (   layerId *       100 ) +
                      cellId;
    rgBuf->insert( chanKey.begin(), chanKey.end(), geometryId );

  }

  return;

}

