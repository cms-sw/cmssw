/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/07/15 15:57:23 $
 *  $Revision: 1.19 $
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
  readOutChannelDriftTubeMap.reserve( 2000 );
}


DTReadOutMapping::DTReadOutMapping( const std::string& cell_map_version,
                                    const std::string&  rob_map_version ):
  cellMapVersion( cell_map_version ),
   robMapVersion(  rob_map_version ) {
  readOutChannelDriftTubeMap.reserve( 2000 );
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
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() +
                                     mapNameGR() );
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() );

/*
  std::string mNrobRG = mapNameRG() + "robMap";
  std::string mNrosRG = mapNameRG() + "rosMap";
  std::string mNdduRG = mapNameRG() + "dduMap";
  DTBufferTree<int,int>* rgROB =
  DTDataBuffer<int,int>::findBuffer( mNrobRG );
  DTBufferTree<int,int>* rgROS =
  DTDataBuffer<int,int>::findBuffer( mNrosRG );
  DTBufferTree<int,int>* rgDDU =
  DTDataBuffer<int,int>::findBuffer( mNdduRG );
  if ( rgROB != 0 ) {
    std::vector<int> rg_list = rgROB->contList();
    std::cout << "drop RG_ROB list " << rg_list.size() << std::endl;
  }
  if ( rgROS != 0 ) {
    std::vector<int> rg_list = rgROS->contList();
    std::cout << "drop RG_ROS list " << rg_list.size() << std::endl;
  }
  if ( rgDDU != 0 ) {
    std::vector<int> rg_list = rgDDU->contList();
    std::cout << "drop RG_DDU list " << rg_list.size() << std::endl;
  }
*/
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() + "robMap" );
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() + "rosMap" );
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() + "dduMap" );
  DTDataBuffer<int,int>::dropBuffer( mapNameGR() );

  std::string mNrobGR = mapNameGR() + "robMap";
  std::string mNrosGR = mapNameGR() + "rosMap";
  std::string mNdduGR = mapNameGR() + "dduMap";
  DTBufferTree<int,std::vector<int>*>* grROB =
  DTDataBuffer<int,std::vector<int>*>::findBuffer( mNrobGR );
  DTBufferTree<int,std::vector<int>*>* grROS =
  DTDataBuffer<int,std::vector<int>*>::findBuffer( mNrosGR );
  DTBufferTree<int,std::vector<int>*>* grDDU =
  DTDataBuffer<int,std::vector<int>*>::findBuffer( mNdduGR );
  if ( grROB != 0 ) {
    std::vector<std::vector<int>*> gr_list = grROB->contList();
    std::vector<std::vector<int>*>::const_iterator gr_iter = gr_list.begin();
    std::vector<std::vector<int>*>::const_iterator gr_iend = gr_list.end();
    while ( gr_iter != gr_iend ) delete *gr_iter++;
//    std::cout << "drop GR_ROB list " << gr_list.size() << std::endl;
  }
  if ( grROS != 0 ) {
    std::vector<std::vector<int>*> gr_list = grROS->contList();
    std::vector<std::vector<int>*>::const_iterator gr_iter = gr_list.begin();
    std::vector<std::vector<int>*>::const_iterator gr_iend = gr_list.end();
    while ( gr_iter != gr_iend ) delete *gr_iter++;
//    std::cout << "drop GR_ROS list " << gr_list.size() << std::endl;
  }
  if ( grDDU != 0 ) {
    std::vector<std::vector<int>*> gr_list = grDDU->contList();
    std::vector<std::vector<int>*>::const_iterator gr_iter = gr_list.begin();
    std::vector<std::vector<int>*>::const_iterator gr_iend = gr_list.end();
    while ( gr_iter != gr_iend ) delete *gr_iter++;
//    std::cout << "drop GR_DDU list " << gr_list.size() << std::endl;
  }

  DTDataBuffer<int,std::vector<int>*>::dropBuffer( mNrobGR );
  DTDataBuffer<int,std::vector<int>*>::dropBuffer( mNrosGR );
  DTDataBuffer<int,std::vector<int>*>::dropBuffer( mNdduGR );
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

  std::string mNameMT = mapNameRG() +
                        mapNameGR();
  DTBufferTree<int,int>* mType =
  DTDataBuffer<int,int>::findBuffer( mNameMT );

  std::string mNameRG = mapNameRG();
  std::string mNrobRG = mapNameRG() + "robMap";
  std::string mNrosRG = mapNameRG() + "rosMap";
  std::string mNdduRG = mapNameRG() + "dduMap";
  DTBufferTree<int,int>* rgBuf =
  DTDataBuffer<int,int>::findBuffer( mNameRG );
  DTBufferTree<int,int>* rgROB =
  DTDataBuffer<int,int>::findBuffer( mNrobRG );
  DTBufferTree<int,int>* rgROS =
  DTDataBuffer<int,int>::findBuffer( mNrosRG );
  DTBufferTree<int,int>* rgDDU =
  DTDataBuffer<int,int>::findBuffer( mNdduRG );

  if ( ( mType == 0 ) ||
       ( rgBuf == 0 ) ||
       ( rgROB == 0 ) ||
       ( rgROS == 0 ) ||
       ( rgDDU == 0 ) ) {
    cacheMap();
    rgBuf =
    DTDataBuffer<int,int>::findBuffer( mNameRG );
    mType =
    DTDataBuffer<int,int>::findBuffer( mNameMT );
    rgROB =
    DTDataBuffer<int,int>::findBuffer( mNrobRG );
    rgROS =
    DTDataBuffer<int,int>::findBuffer( mNrosRG );
    rgDDU =
    DTDataBuffer<int,int>::findBuffer( mNdduRG );
  }

  int defaultValue;
  mType->find( 0, defaultValue );
  if ( defaultValue ) {

    int searchStatus;
    int ientry;

    std::vector<int> dduKey;
    dduKey.reserve( 5 );
    dduKey.push_back( dduId );
    dduKey.push_back( rosId );
//    dduKey.push_back( defaultValue );
//    dduKey.push_back( defaultValue );
//    dduKey.push_back( defaultValue );
//    searchStatus = rgBuf->find( dduKey.begin(), dduKey.end(), ientry );
    searchStatus = rgDDU->find( dduKey.begin(), dduKey.end(), ientry );
    if ( searchStatus ) return searchStatus;
    const DTReadOutGeometryLink& lros( readOutChannelDriftTubeMap[ientry] );
      wheelId = lros.  wheelId;
     sectorId = lros. sectorId;

    std::vector<int> rosKey;
    rosKey.reserve( 5 );
    rosKey.push_back( lros. cellId );
    rosKey.push_back( robId );
    searchStatus = rgROS->find( rosKey.begin(), rosKey.end(), ientry );
    if ( searchStatus ) return searchStatus;
    const DTReadOutGeometryLink& lrob( readOutChannelDriftTubeMap[ientry] );
    if ( lrob.  wheelId != defaultValue )  wheelId = lrob.  wheelId;
    stationId = lrob.stationId;
    if ( lrob. sectorId != defaultValue ) sectorId = lrob. sectorId;

    std::vector<int> robKey;
    robKey.reserve( 5 );
    robKey.push_back( lrob. cellId );
    robKey.push_back(     tdcId );
    robKey.push_back( channelId );
    searchStatus = rgROB->find( robKey.begin(), robKey.end(), ientry );
    if ( searchStatus ) return searchStatus;
    const DTReadOutGeometryLink& ltdc( readOutChannelDriftTubeMap[ientry] );
         slId = ltdc.     slId;
      layerId = ltdc.  layerId;
       cellId = ltdc.   cellId;
    return 0;

  }

  std::vector<int> chanKey;
  chanKey.reserve(5);
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

  std::string mNameMT = mapNameRG() +
                        mapNameGR();
  DTBufferTree<int,int>* mType =
  DTDataBuffer<int,int>::findBuffer( mNameMT );

  std::string mNameGR = mapNameGR();
  std::string mNrobGR = mapNameGR() + "robMap";
  std::string mNrosGR = mapNameGR() + "rosMap";
  std::string mNdduGR = mapNameGR() + "dduMap";
  DTBufferTree<int,int>* grBuf =
  DTDataBuffer<int,int>::findBuffer( mNameGR );
  DTBufferTree<int,std::vector<int>*>* grROB =
  DTDataBuffer<int,std::vector<int>*>::findBuffer( mNrobGR );
  DTBufferTree<int,std::vector<int>*>* grROS =
  DTDataBuffer<int,std::vector<int>*>::findBuffer( mNrosGR );
  DTBufferTree<int,std::vector<int>*>* grDDU =
  DTDataBuffer<int,std::vector<int>*>::findBuffer( mNdduGR );
  if ( ( mType == 0 ) ||
       ( grBuf == 0 ) ||
       ( grROB == 0 ) ||
       ( grROS == 0 ) ||
       ( grDDU == 0 ) ) {
    cacheMap();
    grBuf =
    DTDataBuffer<int,int>::findBuffer( mNameGR );
    mType =
    DTDataBuffer<int,int>::findBuffer( mNameMT );
    grROB =
    DTDataBuffer<int,std::vector<int>*>::findBuffer( mNrobGR );
    grROS =
    DTDataBuffer<int,std::vector<int>*>::findBuffer( mNrosGR );
    grDDU =
    DTDataBuffer<int,std::vector<int>*>::findBuffer( mNdduGR );
  }

  int defaultValue;
  mType->find( 0, defaultValue );
  if ( defaultValue ) {

    int loop1 = 0;
    int loop2 = 0;
    int loop3 = 0;
    int loop0 = 0;

    int searchStatus;
//    int jentry;
    int mapId = 0;
    std::vector<int>* robMLgr;
    std::vector<int>* rosMLgr;
    std::vector<int>* dduMLgr;

    std::vector<int> cellKey;
    cellKey.reserve(6);
    cellKey.push_back(  cellId );
    cellKey.push_back( layerId );
    cellKey.push_back(    slId );
    std::vector<int> stdcKey = cellKey;
//    std::cout << "TDC map: "
//              << stdcKey[0] << " "
//              << stdcKey[1] << " "
//              << stdcKey[2] << " ... " << std::endl;
    searchStatus = grROB->find( cellKey.begin(), cellKey.end(), robMLgr );
    if ( searchStatus ) return searchStatus;
    if ( !( robMLgr->size() ) ) return 1;
    std::vector<int>::const_iterator tdc_iter = robMLgr->begin();
    std::vector<int>::const_iterator tdc_iend = robMLgr->end();
    while( tdc_iter != tdc_iend ) {
      loop1++;
//      jentry = *tdc_iter;
      const DTReadOutGeometryLink& ltdc(
            readOutChannelDriftTubeMap[*tdc_iter++] );
      channelId = ltdc.channelId;
          tdcId = ltdc.    tdcId;
      mapId = ltdc.rosId;
//      std::cout << "TDC map: "
//                << stdcKey[0] << " "
//                << stdcKey[1] << " "
//                << stdcKey[2] << " -> "
//                << mapId << " : "
//                << jentry    <<  " "
//                << channelId <<  " "
//                << tdcId << std::endl;
      cellKey.clear();
      cellKey.push_back( mapId );
      cellKey.push_back( stationId );
      std::vector<int> srosKey = cellKey;
//      std::cout << "ROS map: "
//                << srosKey[0] << " "
//                << srosKey[1] << " ... " << std::endl;
      searchStatus = grROS->find( cellKey.begin(), cellKey.end(), rosMLgr );
      if ( searchStatus ) continue;
      if ( !( rosMLgr->size() ) ) continue;
      std::vector<int>::const_iterator ros_iter = rosMLgr->begin();
      std::vector<int>::const_iterator ros_iend = rosMLgr->end();
      while( ros_iter != ros_iend ) {
        loop2++;
//        jentry = *ros_iter;
        const DTReadOutGeometryLink& lros(
              readOutChannelDriftTubeMap[*ros_iter++] );
        int secCk = lros.sectorId;
        int wheCk = lros. wheelId;
        if ( ( secCk != defaultValue ) && 
             ( secCk != sectorId     ) ) continue;
        if ( ( wheCk != defaultValue ) && 
             ( wheCk !=  wheelId     ) ) continue;
        robId = lros.robId;
        mapId = lros.rosId;
//        std::cout << "ROS map: "
//                  << srosKey[0] << " "
//                  << srosKey[1] << " -> "
//                  << mapId << " : "
//                  << jentry << " "
//                  << robId  << std::endl;
        cellKey.clear();
        cellKey.push_back(    mapId );
        cellKey.push_back(  wheelId );
        cellKey.push_back( sectorId );
        std::vector<int> sdduKey = cellKey;
//        std::cout << "DDU map: "
//                  << grDDU << " "
//                  << sdduKey[0] << " "
//                  << sdduKey[1] << " "
//                  << sdduKey[2] << " ... " << std::endl;
        searchStatus = grDDU->find( cellKey.begin(), cellKey.end(), dduMLgr );
        if ( searchStatus ) continue;
        if ( !( dduMLgr->size() ) ) continue;
        if ( searchStatus ) return searchStatus;
        if ( !( dduMLgr->size() ) ) return 1;
        loop0++;
        std::vector<int>::const_iterator ddu_iter = dduMLgr->begin();
        std::vector<int>::const_iterator ddu_iend = dduMLgr->end();
        while( ddu_iter != ddu_iend ) {
          loop3++;
//          jentry = *ddu_iter;
          const DTReadOutGeometryLink& lddu(
                readOutChannelDriftTubeMap[*ddu_iter++] );
          mapId = lros.rosId;
//          std::cout << "DDU map: "
//                    << sdduKey[0] << " "
//                    << sdduKey[1] << " "
//                    << sdduKey[2] << " -> "
//                    << mapId << " : "
//                    << jentry << std::endl;
          if ( ( ( sectorId == secCk ) || 
                 ( sectorId == lddu.sectorId ) ) &&
               ( (  wheelId == wheCk ) || 
                 (  wheelId == lddu.wheelId ) ) ) {
            rosId = lddu.rosId;
            dduId = lddu.dduId;
//            std::cout << "loop: "
//                      << loop1 << " "
//                      << loop2 << " "
//                      << loop0 << " "
//                      << loop3 << std::endl;
            return 0;
          }
        }
      }
    }

    return 1;

  }

  std::vector<int> cellKey;
  cellKey.reserve(6);
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
  DTDataBuffer<int,int>::dropBuffer( mapNameRG() +
                                     mapNameGR() );
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
  cellKey.reserve(6);
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  int grStatus =
  grBuf->insert( cellKey.begin(), cellKey.end(), ientry );
  std::vector<int> chanKey;
  chanKey.reserve(5);
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
  std::stringstream name;
  name << cellMapVersion << "_" << robMapVersion << "_map_GR" << this;
  return name.str();
}


std::string DTReadOutMapping::mapNameRG() const {
  std::stringstream name;
  name << cellMapVersion << "_" << robMapVersion << "_map_RG" << this;
  return name.str();
}


void DTReadOutMapping::cacheMap() const {

  std::string mNameMT = mapNameRG() +
                        mapNameGR();

  std::string mNameRG = mapNameRG();
  std::string mNrobRG = mapNameRG() + "robMap";
  std::string mNrosRG = mapNameRG() + "rosMap";
  std::string mNdduRG = mapNameRG() + "dduMap";

  std::string mNameGR = mapNameGR();
  std::string mNrobGR = mapNameGR() + "robMap";
  std::string mNrosGR = mapNameGR() + "rosMap";
  std::string mNdduGR = mapNameGR() + "dduMap";

  DTBufferTree<int,int>* mType =
  DTDataBuffer<int,int>::openBuffer( mNameMT );
  mType->insert( 0, 0 );

  DTBufferTree<int,int>* rgBuf =
  DTDataBuffer<int,int>::openBuffer( mNameRG );
  DTBufferTree<int,int>* grBuf =
  DTDataBuffer<int,int>::openBuffer( mNameGR );

  DTBufferTree<int,int>* rgROB =
  DTDataBuffer<int,int>::openBuffer( mNrobRG );
  DTBufferTree<int,int>* rgROS =
  DTDataBuffer<int,int>::openBuffer( mNrosRG );
  DTBufferTree<int,int>* rgDDU =
  DTDataBuffer<int,int>::openBuffer( mNdduRG );
  rgROB->setDefault( 0 );
  rgROS->setDefault( 0 );
  rgDDU->setDefault( 0 );

  DTBufferTree<int,std::vector<int>*>* grROB =
  DTDataBuffer<int,std::vector<int>*>::openBuffer( mNrobGR );
  DTBufferTree<int,std::vector<int>*>* grROS =
  DTDataBuffer<int,std::vector<int>*>::openBuffer( mNrosGR );
  DTBufferTree<int,std::vector<int>*>* grDDU =
  DTDataBuffer<int,std::vector<int>*>::openBuffer( mNdduGR );
  grROB->setDefault( 0 );
  grROS->setDefault( 0 );
  grDDU->setDefault( 0 );

  int entryNum = 0;
  int entryMax = readOutChannelDriftTubeMap.size();
  std::vector<int> cellKey;
  cellKey.reserve(6);
  std::vector<int> chanKey;
  chanKey.reserve(5);
  int defaultValue = 0;
  int key;
  int val;
  int rosMapKey = 0;
  int robMapKey = 0;
  std::map<int,std::vector<int>*> dduEntries;
//  while ( entryNum < entryMax ) {
  for ( entryNum = 0; entryNum < entryMax; entryNum++ ) {

    const DTReadOutGeometryLink& link( readOutChannelDriftTubeMap[entryNum] );

    key = link.dduId;
    val = link.stationId;
    if ( key > 0x3fffffff ) {
      if ( link.  tdcId > 0x3fffffff ) {
        mType->insert( 0, defaultValue = link.  tdcId );
        rosMapKey = key;
      }
      else {
        mType->insert( 0, defaultValue = link.wheelId );
        robMapKey = key;
      }
    }

    if ( defaultValue == 0 ) {

      chanKey.clear();
      chanKey.push_back( link.    dduId );
      chanKey.push_back( link.    rosId );
      chanKey.push_back( link.    robId );
      chanKey.push_back( link.    tdcId );
      chanKey.push_back( link.channelId );

      rgBuf->insert( chanKey.begin(), chanKey.end(), entryNum );

      cellKey.clear();
      cellKey.push_back( link.  wheelId );
      cellKey.push_back( link.stationId );
      cellKey.push_back( link. sectorId );
      cellKey.push_back( link.     slId );
      cellKey.push_back( link.  layerId );
      cellKey.push_back( link.   cellId );

      grBuf->insert( cellKey.begin(), cellKey.end(), entryNum );

    }

    if ( key == robMapKey ) {

      chanKey.clear();
      chanKey.push_back( link.    rosId );
      chanKey.push_back( link.    tdcId );
      chanKey.push_back( link.channelId );
      rgROB->insert( chanKey.begin(), chanKey.end(), entryNum );

      cellKey.clear();
      cellKey.push_back( link. cellId );
      cellKey.push_back( link.layerId );
      cellKey.push_back( link.   slId );
      std::vector<int>* robMLgr;
//      std::cout << "TDC buf: " 
//                << cellKey[0] << " "
//                << cellKey[1] << " "
//                << cellKey[2] << " "
//                << entryNum << std::endl;
      grROB->find( cellKey.begin(), cellKey.end(), robMLgr );
      if ( robMLgr == 0 ) grROB->insert( cellKey.begin(), cellKey.end(),
                                         robMLgr = new std::vector<int> );
      robMLgr->push_back( entryNum );

    }

    if ( key == rosMapKey ) {

      chanKey.clear();
      chanKey.push_back( link.rosId );
      chanKey.push_back( link.robId );
      rgROS->insert( chanKey.begin(), chanKey.end(), entryNum );

      cellKey.clear();
      cellKey.push_back( link.   cellId );
      cellKey.push_back( link.stationId );
      std::vector<int>* rosMLgr;
//      std::cout << "ROS buf: " 
//                << cellKey[0] << " "
//                << cellKey[1] << " "
//                << entryNum << std::endl;
      grROS->find( cellKey.begin(), cellKey.end(), rosMLgr );
      if ( rosMLgr == 0 ) grROS->insert( cellKey.begin(), cellKey.end(),
                                         rosMLgr = new std::vector<int> );
      rosMLgr->push_back( entryNum );

    }

    if ( ( key < 0x3fffffff ) &&
         ( val > 0x3fffffff ) ) {

      chanKey.clear();
      chanKey.push_back( link.dduId );
      chanKey.push_back( link.rosId );
      rgDDU->insert( chanKey.begin(), chanKey.end(), entryNum );

      int mapId = link.cellId;
      std::vector<int>* dduMLgr;
      std::map<int,std::vector<int>*>::const_iterator dduEntIter =
                                       dduEntries.find( mapId );
      if ( dduEntIter == dduEntries.end() )
           dduEntries.insert( std::pair<int,std::vector<int>*>( mapId,
                               dduMLgr = new std::vector<int> ) );
      else                     dduMLgr = dduEntIter->second;
      dduMLgr->push_back( entryNum );

    }

  }

  if ( defaultValue != 0 ) {
//    std::cout << "second loop" << std::endl;
    for ( entryNum = 0; entryNum < entryMax; entryNum++ ) {
      const DTReadOutGeometryLink& link( 
            readOutChannelDriftTubeMap[entryNum] );
      key = link.dduId;
//      std::cout << "key " << key << " " << rosMapKey << std::endl;
      if ( key != rosMapKey ) continue;
      int    mapId = link.   rosId;
      int  whchkId = link. wheelId;
      int secchkId = link.sectorId;

//      std::cout << "check map "
//                << mapId << " "
//                << link.robId << " -> "
//                <<  whchkId << " "
//                << secchkId << std::endl;
      std::vector<int>* dduMLgr;
      std::map<int,std::vector<int>*>::const_iterator dduEntIter =
                                       dduEntries.find( mapId );
      if ( dduEntIter != dduEntries.end() ) dduMLgr = dduEntIter->second;
      else                                  continue;
      std::vector<int>::const_iterator dduIter = dduMLgr->begin();
      std::vector<int>::const_iterator dduIend = dduMLgr->end();
      while( dduIter != dduIend ) {
        int ientry = *dduIter++;
        const DTReadOutGeometryLink& lros(
              readOutChannelDriftTubeMap[ientry] );
        int  wheelId =  whchkId;
        int sectorId = secchkId;
        if (  wheelId == defaultValue )  wheelId = lros. wheelId;
        if ( sectorId == defaultValue ) sectorId = lros.sectorId;
        cellKey.clear();
        cellKey.push_back(    mapId );
        cellKey.push_back(  wheelId );
        cellKey.push_back( sectorId );
//        std::cout <<    mapId << " "
//                  <<  wheelId << " "
//                  << sectorId << " -> "
//                  << ientry << std::endl;
        std::vector<int>* dduMLgr = 0;
        grDDU->find( cellKey.begin(), cellKey.end(), dduMLgr );
        if ( dduMLgr == 0 ) grDDU->insert( cellKey.begin(), cellKey.end(),
                                           dduMLgr = new std::vector<int> );
        dduMLgr->push_back( ientry );
      }

    }

    DTDataBuffer<int,int>::dropBuffer( mNameRG );
    DTDataBuffer<int,int>::dropBuffer( mNameGR );
    DTDataBuffer<int,int>::openBuffer( mNameRG );
    DTDataBuffer<int,int>::openBuffer( mNameGR );

  }

  return;

}

