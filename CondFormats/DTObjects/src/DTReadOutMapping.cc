/*
 *  See header file for a description of this class.
 *
 *  $Date: 2005/11/15 13:50:55 $
 *  $Revision: 1.2 $
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
//  std::cout << "constructor 0: -"
//            << cellMapVersion << "- -"
//            <<  robMapVersion << "-" << std::endl;
}

DTReadOutMapping::DTReadOutMapping( const std::string& cell_map_version,
                                    const std::string&  rob_map_version ):
 cellMapVersion( cell_map_version ),
  robMapVersion(  rob_map_version ) {

//  std::cout << "constructor 2: -"
//            << cellMapVersion << "- -"
//            <<  robMapVersion << "-" << std::endl;

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
//  std::cout << "initSetup: " << mappingVersion << std::endl;
  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
//  DTDataBuffer<int>::insertTDCChannelData( mappingVersion,
//                                           1, 1, 1, 1, 1, -999, -999 );
//  DTDataBuffer<int>::insertCellData( mappingVersion,
//                                        1, 1, 1, 1, 1, 1, -999, -999 );
  DTDataBuffer<int>::openBuffer( "tdc_channel", mappingVersion, -999 );
  DTDataBuffer<int>::openBuffer(        "cell", mappingVersion, -999 );
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
                                             link.   cellId,
                                             -999 );
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
                                       link.channelId,
                                       -999 );
  }

//  std::cout << "initSetup - end " << std::endl;

  return;

}


DTDetId DTReadOutMapping::readOutToGeometry( int      dduId,
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

  return DTDetId( wheelId, stationId, sectorId, slId, layerId, cellId );

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

//  int found =
  wheelId   =
  stationId =
  sectorId  =
  slId      =
  layerId   =
  cellId    = 0;

/*
  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
  while ( iter != iend ) {
    const DTReadOutGeometryLink& link = *iter++;
    if ( link.    dduId !=     dduId ) continue;
    if ( link.    rosId !=     rosId ) continue;
    if ( link.    robId !=     robId ) continue;
    if ( link.    tdcId !=     tdcId ) continue;
    if ( link.channelId != channelId ) continue;
      wheelId = link.  wheelId;
    stationId = link.stationId;
     sectorId = link. sectorId;
         slId = link.     slId;
      layerId = link.  layerId;
       cellId = link.   cellId;
    found = 1;
  }
*/

  std::string mappingVersion = cellMapVersion + "_" + robMapVersion + "_map";
  if( DTDataBuffer<int>::findBuffer( "tdc_channel",
                                     mappingVersion ) == 0 ) initSetup();
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

  return; //found;

}


//int DTReadOutMapping::geometryToReadOut( int    wheelId,
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

//  int found =
  dduId =
  rosId =
  robId =
  tdcId =
  channelId = 0;

/*
  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
  while ( iter != iend ) {
    const DTReadOutGeometryLink& link = *iter++;
    if ( link.  wheelId !=   wheelId ) continue;
    if ( link.stationId != stationId ) continue;
    if ( link. sectorId !=  sectorId ) continue;
    if ( link.     slId !=      slId ) continue;
    if ( link.  layerId !=   layerId ) continue;
    if ( link.   cellId !=    cellId ) continue;
        dduId = link.    dduId;
        rosId = link.    rosId;
        robId = link.    robId;
        tdcId = link.    tdcId;
    channelId = link.channelId;
    found = 1;
  }
*/

  std::string mappingVersion = cellMapVersion + "_" + robMapVersion + "_map";
  if( DTDataBuffer<int>::findBuffer( "cell",
                                     mappingVersion ) == 0 ) initSetup();
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
      dduId = readoutId %  100;

  return; //found;

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

  std::vector<DTReadOutGeometryLink>::const_iterator iter =
              readOutChannelDriftTubeMap.begin();
  std::vector<DTReadOutGeometryLink>::const_iterator iend =
              readOutChannelDriftTubeMap.end();
  bool exist = false;
  while ( iter != iend ) {
    const DTReadOutGeometryLink& link = *iter++;
    exist = true;
    if ( ( link.    dduId ==     dduId ) &&
         ( link.    rosId ==     rosId ) &&
         ( link.    robId ==     robId ) &&
         ( link.    tdcId ==     tdcId ) &&
         ( link.channelId == channelId ) ) break;
    if ( ( link.  wheelId ==   wheelId ) &&
         ( link.stationId == stationId ) &&
         ( link. sectorId ==  sectorId ) &&
         ( link.     slId ==      slId ) &&
         ( link.  layerId ==   layerId ) &&
         ( link.   cellId ==    cellId ) ) break;
    exist = false;
  }
  if ( exist ) return 1;

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
					   cellId,
                                           -999 );
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
                                     channelId,
                                     -999 );

  return 0;

}


DTReadOutMapping::const_iterator DTReadOutMapping::begin() const {
  return readOutChannelDriftTubeMap.begin();
}


DTReadOutMapping::const_iterator DTReadOutMapping::end() const {
  return readOutChannelDriftTubeMap.end();
}


