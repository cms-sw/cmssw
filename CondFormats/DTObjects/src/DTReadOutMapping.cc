/*
 *  See header file for a description of this class.
 *
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
#include "CondFormats/DTObjects/interface/DTBufferTree.h"
#include "CondFormats/DTObjects/interface/DTCompactMapAbstractHandler.h"
#include "CondFormats/DTObjects/interface/DTReadOutMappingCache.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <sstream>
#include <map>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTReadOutMapping::DTReadOutMapping():
  cellMapVersion( " " ),
  robMapVersion( " " ),
  rgBuf(new DTBufferTree<int,int>),
  grBuf(new DTBufferTree<int,int>) {

  readOutChannelDriftTubeMap.reserve( 2000 );
}


DTReadOutMapping::DTReadOutMapping( const std::string& cell_map_version,
                                    const std::string&  rob_map_version ):
  cellMapVersion( cell_map_version ),
  robMapVersion(  rob_map_version ),
  rgBuf(new DTBufferTree<int,int>),
  grBuf(new DTBufferTree<int,int>) {

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

  DTBufferTree<int,int> const* pmType;
  DTBufferTree<int,int> const* prgBuf;
  DTBufferTree<int,int> const* prgDDU;
  DTBufferTree<int,int> const* prgROS;
  DTBufferTree<int,int> const* prgROB;

  if(cache().isSet()) {
    pmType = cache()->mType.get();
    prgBuf = cache()->rgBuf.get();
    prgDDU = cache()->rgDDU.get();
    prgROS = cache()->rgROS.get();
    prgROB = cache()->rgROB.get();
  } else {
    if(!atomicCache().isSet()) {
      cacheMap();
    }
    pmType = atomicCache()->mType.get();
    prgBuf = atomicCache()->rgBuf.get();
    prgDDU = atomicCache()->rgDDU.get();
    prgROS = atomicCache()->rgROS.get();
    prgROB = atomicCache()->rgROB.get();
  }

  int defaultValue;
  pmType->find( 0, defaultValue );
  if ( defaultValue ) {

    int searchStatus;
    int ientry;

    std::vector<int> dduKey;
    dduKey.reserve( 5 );
    dduKey.push_back( dduId );
    dduKey.push_back( rosId );
    searchStatus = prgDDU->find( dduKey.begin(), dduKey.end(), ientry );
    if ( searchStatus ) return searchStatus;
    const DTReadOutGeometryLink& lros( readOutChannelDriftTubeMap[ientry] );
    wheelId = lros.  wheelId;
    sectorId = lros. sectorId;

    std::vector<int> rosKey;
    rosKey.reserve( 5 );
    rosKey.push_back( lros. cellId );
    rosKey.push_back( robId );
    searchStatus = prgROS->find( rosKey.begin(), rosKey.end(), ientry );
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
    searchStatus = prgROB->find( robKey.begin(), robKey.end(), ientry );
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
  int searchStatus = prgBuf->find( chanKey.begin(), chanKey.end(), ientry );
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

  DTBufferTree<int,int> const* pmType;
  DTBufferTree<int,int> const* pgrBuf;
  DTBufferTreeUniquePtr const* pgrDDU;
  DTBufferTreeUniquePtr const* pgrROS;
  DTBufferTreeUniquePtr const* pgrROB;

  if(cache().isSet()) {
    pmType = cache()->mType.get();
    pgrBuf = cache()->grBuf.get();
    pgrDDU = cache()->grDDU.get();
    pgrROS = cache()->grROS.get();
    pgrROB = cache()->grROB.get();
  } else {
    if(!atomicCache().isSet()) {
      cacheMap();
    }
    pmType = atomicCache()->mType.get();
    pgrBuf = atomicCache()->grBuf.get();
    pgrDDU = atomicCache()->grDDU.get();
    pgrROS = atomicCache()->grROS.get();
    pgrROB = atomicCache()->grROB.get();
  }

  int defaultValue;
  pmType->find( 0, defaultValue );
  if ( defaultValue ) {

    int loop1 = 0;
    int loop2 = 0;
    int loop3 = 0;
    int loop0 = 0;

    int searchStatus;
    int mapId = 0;
    std::vector<int> const* robMLgr;
    std::vector<int> const* rosMLgr;
    std::vector<int> const* dduMLgr;

    std::vector<int> cellKey;
    cellKey.reserve(6);
    cellKey.push_back(  cellId );
    cellKey.push_back( layerId );
    cellKey.push_back(    slId );
    std::vector<int> stdcKey = cellKey;
    searchStatus = pgrROB->find( cellKey.begin(), cellKey.end(), robMLgr );
    if ( searchStatus ) return searchStatus;
    if ( !( robMLgr->size() ) ) return 1;
    std::vector<int>::const_iterator tdc_iter = robMLgr->begin();
    std::vector<int>::const_iterator tdc_iend = robMLgr->end();
    while( tdc_iter != tdc_iend ) {
      loop1++;
      const DTReadOutGeometryLink& ltdc(
            readOutChannelDriftTubeMap[*tdc_iter++] );
      channelId = ltdc.channelId;
          tdcId = ltdc.    tdcId;
      mapId = ltdc.rosId;
      cellKey.clear();
      cellKey.push_back( mapId );
      cellKey.push_back( stationId );
      std::vector<int> srosKey = cellKey;
      searchStatus = pgrROS->find( cellKey.begin(), cellKey.end(), rosMLgr );
      if ( searchStatus ) continue;
      if ( !( rosMLgr->size() ) ) continue;
      std::vector<int>::const_iterator ros_iter = rosMLgr->begin();
      std::vector<int>::const_iterator ros_iend = rosMLgr->end();
      while( ros_iter != ros_iend ) {
        loop2++;
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
        cellKey.clear();
        cellKey.push_back(    mapId );
        cellKey.push_back(  wheelId );
        cellKey.push_back( sectorId );
        std::vector<int> sdduKey = cellKey;
        searchStatus = pgrDDU->find( cellKey.begin(), cellKey.end(), dduMLgr );
        if ( searchStatus ) continue;
        if ( !( dduMLgr->size() ) ) continue;
        if ( searchStatus ) return searchStatus;
        if ( !( dduMLgr->size() ) ) return 1;
        loop0++;
        std::vector<int>::const_iterator ddu_iter = dduMLgr->begin();
        std::vector<int>::const_iterator ddu_iend = dduMLgr->end();
        while( ddu_iter != ddu_iend ) {
          loop3++;
          const DTReadOutGeometryLink& lddu(
                readOutChannelDriftTubeMap[*ddu_iter++] );
          mapId = lros.rosId;
          if ( ( ( sectorId == secCk ) || 
                 ( sectorId == lddu.sectorId ) ) &&
               ( (  wheelId == wheCk ) || 
                 (  wheelId == lddu.wheelId ) ) ) {
            rosId = lddu.rosId;
            dduId = lddu.dduId;
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
  int searchStatus = pgrBuf->find( cellKey.begin(), cellKey.end(), ientry );
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


DTReadOutMapping::type DTReadOutMapping::mapType() const {

  DTBufferTree<int,int> const* pmType;
  if(cache().isSet()) {
    pmType = cache()->mType.get();
  } else {
    if(!atomicCache().isSet()) {
      cacheMap();
    }
    pmType = atomicCache()->mType.get();
  }

  int defaultValue;
  pmType->find( 0, defaultValue );
  if ( defaultValue ) return compact;
  else                return plain;
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
  cache().reset();
  atomicCache().reset();
  rgBuf->clear();
  grBuf->clear();
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

  DTBufferTree<int,int>* pgrBuf;
  DTBufferTree<int,int>* prgBuf;

  if(cache().isSet()) {
    pgrBuf = cache()->grBuf.get();
    prgBuf = cache()->rgBuf.get();
  } else if(atomicCache().isSet()) {
    pgrBuf = atomicCache()->grBuf.get();
    prgBuf = atomicCache()->rgBuf.get();
  } else {
    pgrBuf = grBuf.get();
    prgBuf = rgBuf.get();
  }

  std::vector<int> cellKey;
  cellKey.reserve(6);
  cellKey.push_back(   wheelId );
  cellKey.push_back( stationId );
  cellKey.push_back(  sectorId );
  cellKey.push_back(      slId );
  cellKey.push_back(   layerId );
  cellKey.push_back(    cellId );
  int grStatus = pgrBuf->insert( cellKey.begin(), cellKey.end(), ientry );

  std::vector<int> chanKey;
  chanKey.reserve(5);
  chanKey.push_back(     dduId );
  chanKey.push_back(     rosId );
  chanKey.push_back(     robId );
  chanKey.push_back(     tdcId );
  chanKey.push_back( channelId );
  int rgStatus = prgBuf->insert( chanKey.begin(), chanKey.end(), ientry );

  if ( grStatus || rgStatus ) return 1;
  else                        return 0;

}


DTReadOutMapping::const_iterator DTReadOutMapping::begin() const {
  return readOutChannelDriftTubeMap.begin();
}


DTReadOutMapping::const_iterator DTReadOutMapping::end() const {
  return readOutChannelDriftTubeMap.end();
}


const DTReadOutMapping* DTReadOutMapping::fullMap() const {
  if ( mapType() == plain ) return this;
  DTCompactMapAbstractHandler* cmHandler =
                               DTCompactMapAbstractHandler::getInstance();
  if ( cmHandler == 0 ) {
      std::cout << "CondCoreDTPlugins library not loaded, "
                << "compactMapHandler plugin missing" << std::endl;
    return 0;
  }
  return cmHandler->expandMap( *this );
}


void DTReadOutMapping::initialize() {
  if(cache().isSet()) {
    return;
  } else if(atomicCache().isSet()) {
    cache().set(std::unique_ptr<DTReadOutMappingCache>(atomicCache().release()));
  } else {
    cacheMap();
    cache().set(std::unique_ptr<DTReadOutMappingCache>(atomicCache().release()));
  }
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

  std::unique_ptr<DTReadOutMappingCache> theCache(new DTReadOutMappingCache);

  theCache->mType.set(std::unique_ptr<DTBufferTree<int,int> >(new DTBufferTree<int,int>));

  theCache->mType->insert( 0, 0 );

  theCache->rgBuf.set(std::unique_ptr<DTBufferTree<int,int> >(new DTBufferTree<int,int>));
  theCache->rgROB.set(std::unique_ptr<DTBufferTree<int,int> >(new DTBufferTree<int,int>));
  theCache->rgROS.set(std::unique_ptr<DTBufferTree<int,int> >(new DTBufferTree<int,int>));
  theCache->rgDDU.set(std::unique_ptr<DTBufferTree<int,int> >(new DTBufferTree<int,int>));
  theCache->grBuf.set(std::unique_ptr<DTBufferTree<int,int> >(new DTBufferTree<int,int>));

  theCache->grROB.set(std::unique_ptr<DTBufferTreeUniquePtr>(new DTBufferTreeUniquePtr));
  theCache->grROS.set(std::unique_ptr<DTBufferTreeUniquePtr>(new DTBufferTreeUniquePtr));
  theCache->grDDU.set(std::unique_ptr<DTBufferTreeUniquePtr>(new DTBufferTreeUniquePtr));

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
  for ( entryNum = 0; entryNum < entryMax; entryNum++ ) {

    const DTReadOutGeometryLink& link( readOutChannelDriftTubeMap[entryNum] );

    key = link.dduId;
    val = link.stationId;
    if ( key > 0x3fffffff ) {
      if ( link.  tdcId > 0x3fffffff ) {
        theCache->mType->insert( 0, defaultValue = link.  tdcId );
        rosMapKey = key;
      }
      else {
        theCache->mType->insert( 0, defaultValue = link.wheelId );
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

      theCache->rgBuf->insert( chanKey.begin(), chanKey.end(), entryNum );

      cellKey.clear();
      cellKey.push_back( link.  wheelId );
      cellKey.push_back( link.stationId );
      cellKey.push_back( link. sectorId );
      cellKey.push_back( link.     slId );
      cellKey.push_back( link.  layerId );
      cellKey.push_back( link.   cellId );

      theCache->grBuf->insert( cellKey.begin(), cellKey.end(), entryNum );

    }

    if ( key == robMapKey ) {

      chanKey.clear();
      chanKey.push_back( link.    rosId );
      chanKey.push_back( link.    tdcId );
      chanKey.push_back( link.channelId );
      theCache->rgROB->insert( chanKey.begin(), chanKey.end(), entryNum );

      cellKey.clear();
      cellKey.push_back( link. cellId );
      cellKey.push_back( link.layerId );
      cellKey.push_back( link.   slId );
      std::vector<int>* robMLgr;
      theCache->grROB->find( cellKey.begin(), cellKey.end(), robMLgr );
      if ( robMLgr == 0 ) {
        std::unique_ptr<std::vector<int> > newVector(new std::vector<int>);
        robMLgr = newVector.get();
        theCache->grROB->insert( cellKey.begin(), cellKey.end(),
                                 std::move(newVector));
      }
      robMLgr->push_back( entryNum );
    }

    if ( key == rosMapKey ) {

      chanKey.clear();
      chanKey.push_back( link.rosId );
      chanKey.push_back( link.robId );
      theCache->rgROS->insert( chanKey.begin(), chanKey.end(), entryNum );

      cellKey.clear();
      cellKey.push_back( link.   cellId );
      cellKey.push_back( link.stationId );
      std::vector<int>* rosMLgr;
      theCache->grROS->find( cellKey.begin(), cellKey.end(), rosMLgr );
      if ( rosMLgr == 0 ) {
        std::unique_ptr<std::vector<int> > newVector(new std::vector<int>);
        rosMLgr = newVector.get();
        theCache->grROS->insert( cellKey.begin(), cellKey.end(),
                                 std::move(newVector));
      }
      rosMLgr->push_back( entryNum );
    }

    if ( ( key < 0x3fffffff ) &&
         ( val > 0x3fffffff ) ) {

      chanKey.clear();
      chanKey.push_back( link.dduId );
      chanKey.push_back( link.rosId );
      theCache->rgDDU->insert( chanKey.begin(), chanKey.end(), entryNum );

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
    for ( entryNum = 0; entryNum < entryMax; entryNum++ ) {
      const DTReadOutGeometryLink& link( 
            readOutChannelDriftTubeMap[entryNum] );
      key = link.dduId;
      if ( key != rosMapKey ) continue;
      int    mapId = link.   rosId;
      int  whchkId = link. wheelId;
      int secchkId = link.sectorId;

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
        std::vector<int>* dduMLgr = 0;
        theCache->grDDU->find( cellKey.begin(), cellKey.end(), dduMLgr );
        if ( dduMLgr == 0 ) {
          std::unique_ptr<std::vector<int> > newVector(new std::vector<int>);
          dduMLgr = newVector.get();
          theCache->grDDU->insert( cellKey.begin(), cellKey.end(),
                                   std::move(newVector) );
        }
        dduMLgr->push_back( ientry );
      }
    }

    std::map<int,std::vector<int>*>::const_iterator dduEntIter =
                                                    dduEntries.begin();
    std::map<int,std::vector<int>*>::const_iterator dduEntIend =
                                                    dduEntries.end();
    while ( dduEntIter != dduEntIend ) {
      const std::pair<int,std::vector<int>*>& dduEntry = *dduEntIter++;
      delete dduEntry.second;
    }

    theCache->rgBuf->clear();
    theCache->grBuf->clear();
  }

  atomicCache().set(std::move(theCache));

  return;
}
