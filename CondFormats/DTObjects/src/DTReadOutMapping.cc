/*
 *  See header file for a description of this class.
 *
 *  $Date: 2006/01/27 15:22:15 $
 *  $Revision: 1.5 $
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

  initSetup();

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
}

DTReadOutGeometryLink::~DTReadOutGeometryLink() {
}

//--------------
// Operations --
//--------------
void DTReadOutMapping::initSetup() const {

  std::string mappingVersion = cellMapVersion + "_" + robMapVersion + "_map";
  int minWheel;
  int minStation;
  int minSector;
  int minSL;
  int minLayer;
  int minCell;
  int minDDU;
  int minROS;
  int minROB;
  int minTDC;
  int minChannel;
  int maxWheel;
  int maxStation;
  int maxSector;
  int maxSL;
  int maxLayer;
  int maxCell;
  int maxDDU;
  int maxROS;
  int maxROB;
  int maxTDC;
  int maxChannel;

  getIdNumbers( minWheel, minStation, minSector, minSL,  minLayer,   minCell,
                minDDU,   minROS,     minROB,    minTDC, minChannel,
                maxWheel, maxStation, maxSector, maxSL,  maxLayer,   maxCell,
                maxDDU,   maxROS,     maxROB,    maxTDC, maxChannel );

  DTDataBuffer<int>::openBuffer( "tdc_channel", mappingVersion,
                minDDU,   minROS,     minROB,    minTDC, minChannel, 0,
                maxDDU,   maxROS,     maxROB,    maxTDC, maxChannel, 1,
                -999 );
  DTDataBuffer<int>::openBuffer(        "cell", mappingVersion,
                minWheel, minStation, minSector, minSL,  minLayer,   minCell,
                maxWheel, maxStation, maxSector, maxSL,  maxLayer,   maxCell,
                -999 );

  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
  while ( iter != iend ) {
    const DTReadOutGeometryLink& link = *iter++;
    DTDataBuffer<int>::insertTDCChannelData( mappingVersion,
                                             link.    dduId,
                                             link.    rosId,
                                             link.    robId,
                                             link.    tdcId,
                                             link.channelId,
                                           ( link.  wheelId *  10000000 ) +
                                           ( link.stationId *   1000000 ) +
                                           ( link. sectorId *     10000 ) +
                                           ( link.     slId *      1000 ) +
                                           ( link.  layerId *       100 ) +
                                             link.   cellId );
    DTDataBuffer<int>::insertCellData( mappingVersion,
                                       link.  wheelId,
                                       link.stationId,
                                       link. sectorId,
                                       link.     slId,
                                       link.  layerId,
                                       link.   cellId,
                                     ( link.    dduId *  1000000 ) +
                                     ( link.    rosId *    10000 ) +
                                     ( link.    robId *     1000 ) +
                                     ( link.    tdcId *      100 ) +
                                       link.channelId );
  }

  return;

}


DTWireId DTReadOutMapping::readOutToGeometry( int      dduId,
                                              int      rosId,
                                              int      robId,
                                              int      tdcId,
                                              int  channelId ) const {

  int   wheelId;
  int stationId;
  int  sectorId;
  int      slId;
  int   layerId;
  int    cellId;

  readOutToGeometry(      dduId,
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

  return DTWireId( wheelId, stationId, sectorId, slId, layerId, cellId );

}

void DTReadOutMapping::readOutToGeometry( int      dduId,
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


  std::string mappingVersion = cellMapVersion + "_" + robMapVersion + "_map";
  if( !DTDataBuffer<int>::findBuffer( "tdc_channel",
                                     mappingVersion ) ) initSetup();
  int geometryId =
  DTDataBuffer<int>::getTDCChannelData( mappingVersion,
                                            dduId,
                                            rosId,
                                            robId,
                                            tdcId,
                                        channelId );

  if ( geometryId == -999 ) return;
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

  return;

}


void DTReadOutMapping::geometryToReadOut( int    wheelId,
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

  std::string mappingVersion = cellMapVersion + "_" + robMapVersion + "_map";
  if( !DTDataBuffer<int>::findBuffer( "cell",
                                     mappingVersion ) ) initSetup();
  int readoutId =
  DTDataBuffer<int>::getCellData( mappingVersion,
                                         wheelId,
                                       stationId,
                                        sectorId,
                                            slId,
                                         layerId,
                                          cellId );

  if ( readoutId == -999 ) return;
  channelId = readoutId %  100;
              readoutId /= 100;
      tdcId = readoutId %  10;
              readoutId /= 10;
      robId = readoutId %  10;
              readoutId /= 10;
      rosId = readoutId %  100;
              readoutId /= 100;
      dduId = readoutId %  1000;

  return;

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

  std::string mappingVersion = cellMapVersion + "_" + robMapVersion + "_map";
  if( !DTDataBuffer<int>::findBuffer( "cell", mappingVersion ) ) return 0;
  return 0;
  DTDataBuffer<int>::insertTDCChannelData( mappingVersion,
                                               dduId,
                                               rosId,
                                               robId,
                                               tdcId,
                                           channelId,
                                         (   wheelId *  10000000 ) +
                                         ( stationId *   1000000 ) +
                                         (  sectorId *     10000 ) +
                                         (      slId *      1000 ) +
                                         (   layerId *       100 ) +
					   cellId );
  DTDataBuffer<int>::insertCellData( mappingVersion,
                                       wheelId,
                                     stationId,
                                      sectorId,
                                          slId,
                                       layerId,
                                        cellId,
                                   (     dduId *  1000000 ) +
                                   (     rosId *    10000 ) +
                                   (     robId *     1000 ) +
                                   (     tdcId *      100 ) +
                                     channelId );

  return 0;

}


DTReadOutMapping::const_iterator DTReadOutMapping::begin() const {
  return readOutChannelDriftTubeMap.begin();
}


DTReadOutMapping::const_iterator DTReadOutMapping::end() const {
  return readOutChannelDriftTubeMap.end();
}


void DTReadOutMapping::getIdNumbers( int& minWheel,   int& minStation,
                                     int& minSector,  int& minSL,
                                     int& minLayer,   int& minCell,
                                     int& minDDU,     int& minROS,
                                     int& minROB,     int& minTDC,
                                     int& minChannel,
                                     int& maxWheel,   int& maxStation,
                                     int& maxSector,  int& maxSL,
                                     int& maxLayer,   int& maxCell,
                                     int& maxDDU,     int& maxROS,
                                     int& maxROB,     int& maxTDC,
                                     int& maxChannel ) const {

  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
  minWheel   = 99999999;
  minStation = 99999999;
  minSector  = 99999999;
  minSL      = 99999999;
  minLayer   = 99999999;
  minCell    = 99999999;
  minDDU     = 99999999;
  minROS     = 99999999;
  minROB     = 99999999;
  minTDC     = 99999999;
  minChannel = 99999999;
  maxWheel   = 0;
  maxStation = 0;
  maxSector  = 0;
  maxSL      = 0;
  maxLayer   = 0;
  maxCell    = 0;
  maxDDU     = 0;
  maxROS     = 0;
  maxROB     = 0;
  maxTDC     = 0;
  maxChannel = 0;
  int id;
  int nfound = 0;
  while ( iter != iend ) {
    const DTReadOutGeometryLink& link = *iter++;
    if ( ( id = link.  wheelId ) < minWheel   ) minWheel   = id;
    if ( ( id = link.stationId ) < minStation ) minStation = id;
    if ( ( id = link. sectorId ) < minSector  ) minSector  = id;
    if ( ( id = link.     slId ) < minSL      ) minSL      = id;
    if ( ( id = link.  layerId ) < minLayer   ) minLayer   = id;
    if ( ( id = link.   cellId ) < minCell    ) minCell    = id;
    if ( ( id = link.    dduId ) < minDDU     ) minDDU     = id;
    if ( ( id = link.    rosId ) < minROS     ) minROS     = id;
    if ( ( id = link.    robId ) < minROB     ) minROB     = id;
    if ( ( id = link.    tdcId ) < minTDC     ) minTDC     = id;
    if ( ( id = link.channelId ) < minChannel ) minChannel = id;
    if ( ( id = link.  wheelId ) > maxWheel   ) maxWheel   = id;
    if ( ( id = link.stationId ) > maxStation ) maxStation = id;
    if ( ( id = link. sectorId ) > maxSector  ) maxSector  = id;
    if ( ( id = link.     slId ) > maxSL      ) maxSL      = id;
    if ( ( id = link.  layerId ) > maxLayer   ) maxLayer   = id;
    if ( ( id = link.   cellId ) > maxCell    ) maxCell    = id;
    if ( ( id = link.    dduId ) > maxDDU     ) maxDDU     = id;
    if ( ( id = link.    rosId ) > maxROS     ) maxROS     = id;
    if ( ( id = link.    robId ) > maxROB     ) maxROB     = id;
    if ( ( id = link.    tdcId ) > maxTDC     ) maxTDC     = id;
    if ( ( id = link.channelId ) > maxChannel ) maxChannel = id;
    nfound++;
  }

  if ( nfound == 0 ) {
    minWheel   = 1;
    minStation = 1;
    minSector  = 1;
    minSL      = 1;
    minLayer   = 1;
    minCell    = 1;
    minDDU     = 1;
    minROS     = 1;
    minROB     = 1;
    minTDC     = 1;
    minChannel = 1;
    maxWheel   = 0;
    maxStation = 0;
    maxSector  = 0;
    maxSL      = 0;
    maxLayer   = 0;
    maxCell    = 0;
    maxDDU     = 0;
    maxROS     = 0;
    maxROB     = 0;
    maxTDC     = 0;
    maxChannel = 0;
  }

  return;

}

