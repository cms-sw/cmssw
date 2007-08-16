/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/03/06 14:30:40 $
 *  $Revision: 1.14 $
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
  if ( !status ) wireId = DTWireId( wheelId, stationId, sectorId,
                                       slId,   layerId,   cellId );
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
  DTBufferTree<int,const DTReadOutGeometryLink*>* rgBuf =
  DTDataBuffer<int,const DTReadOutGeometryLink*>::findBuffer( mapRtoG );
  if ( rgBuf == 0 ) {
    initSetup();
    rgBuf =
    DTDataBuffer<int,const DTReadOutGeometryLink*>::findBuffer( mapRtoG );
  }
  std::vector<int> chanKey;
  chanKey.push_back(     dduId );
  chanKey.push_back(     rosId );
  chanKey.push_back(     robId );
  chanKey.push_back(     tdcId );
  chanKey.push_back( channelId );
  const DTReadOutGeometryLink* link;
  int searchStatus = rgBuf->find( chanKey.begin(), chanKey.end(), link );
  if ( !searchStatus ) {
      wheelId = link->  wheelId;
    stationId = link->stationId;
     sectorId = link-> sectorId;
         slId = link->     slId;
      layerId = link->  layerId;
       cellId = link->   cellId;
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
  DTBufferTree<int,const DTReadOutGeometryLink*>* grBuf =
  DTDataBuffer<int,const DTReadOutGeometryLink*>::findBuffer( mapGtoR );
  if ( grBuf == 0 ) {
    initSetup();
    grBuf =
    DTDataBuffer<int,const DTReadOutGeometryLink*>::findBuffer( mapGtoR );
  }
  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  const DTReadOutGeometryLink* link;
  int searchStatus = grBuf->find( cellKey.begin(), cellKey.end(), link );
  if ( !searchStatus ) {
        dduId = link->    dduId;
        rosId = link->    rosId;
        robId = link->    robId;
        tdcId = link->    tdcId;
    channelId = link->channelId;
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
  const DTReadOutGeometryLink* lptr = &( readOutChannelDriftTubeMap.back() );

  std::string mapRtoG =
       cellMapVersion + "_" + robMapVersion + "_map_RG";
  std::string mapGtoR =
       cellMapVersion + "_" + robMapVersion + "_map_GR";

  DTBufferTree<int,const DTReadOutGeometryLink*>* rgBuf =
  DTDataBuffer<int,const DTReadOutGeometryLink*>::openBuffer( mapRtoG );
  DTBufferTree<int,const DTReadOutGeometryLink*>* grBuf =
  DTDataBuffer<int,const DTReadOutGeometryLink*>::openBuffer( mapGtoR );

  std::vector<int> cellKey;
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  int grStatus =
  grBuf->insert( cellKey.begin(), cellKey.end(), lptr );
  std::vector<int> chanKey;
  chanKey.push_back(     dduId );
  chanKey.push_back(     rosId );
  chanKey.push_back(     robId );
  chanKey.push_back(     tdcId );
  chanKey.push_back( channelId );
  int rgStatus =
  rgBuf->insert( chanKey.begin(), chanKey.end(), lptr );

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

  DTBufferTree<int,const DTReadOutGeometryLink*>* rgBuf =
  DTDataBuffer<int,const DTReadOutGeometryLink*>::openBuffer( mapRtoG );
  DTBufferTree<int,const DTReadOutGeometryLink*>* grBuf =
  DTDataBuffer<int,const DTReadOutGeometryLink*>::openBuffer( mapGtoR );

  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
  int      dduId;
  int      rosId;
  int      robId;
  int      tdcId;
  int  channelId;
  int    wheelId;
  int  stationId;
  int   sectorId;
  int       slId;
  int    layerId;
  int     cellId;
  while ( iter != iend ) {

    const DTReadOutGeometryLink& link = *iter++;
        dduId = link.    dduId;
        rosId = link.    rosId;
        robId = link.    robId;
        tdcId = link.    tdcId;
    channelId = link.channelId;
      wheelId = link.  wheelId;
    stationId = link.stationId;
     sectorId = link. sectorId;
         slId = link.     slId;
      layerId = link.  layerId;
       cellId = link.   cellId;

    const DTReadOutGeometryLink* lptr = &link;

    std::vector<int> cellKey;
    cellKey.push_back(   wheelId );
    cellKey.push_back( stationId );
    cellKey.push_back(  sectorId );
    cellKey.push_back(      slId );
    cellKey.push_back(   layerId );
    cellKey.push_back(    cellId );

    grBuf->insert( cellKey.begin(), cellKey.end(), lptr );

    std::vector<int> chanKey;
    chanKey.push_back(     dduId );
    chanKey.push_back(     rosId );
    chanKey.push_back(     robId );
    chanKey.push_back(     tdcId );
    chanKey.push_back( channelId );

    rgBuf->insert( chanKey.begin(), chanKey.end(), lptr );

  }

  return;

}

