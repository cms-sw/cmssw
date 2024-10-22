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
DTReadOutMapping::DTReadOutMapping()
    : cellMapVersion(" "), robMapVersion(" "), rgBuf(new DTBufferTree<int, int>), grBuf(new DTBufferTree<int, int>) {
  readOutChannelDriftTubeMap.reserve(2000);
}

DTReadOutMapping::DTReadOutMapping(const std::string& cell_map_version, const std::string& rob_map_version)
    : cellMapVersion(cell_map_version),
      robMapVersion(rob_map_version),
      rgBuf(new DTBufferTree<int, int>),
      grBuf(new DTBufferTree<int, int>) {
  readOutChannelDriftTubeMap.reserve(2000);
}

DTReadOutGeometryLink::DTReadOutGeometryLink()
    : dduId(0),
      rosId(0),
      robId(0),
      tdcId(0),
      channelId(0),
      wheelId(0),
      stationId(0),
      sectorId(0),
      slId(0),
      layerId(0),
      cellId(0) {}

//--------------
// Destructor --
//--------------
DTReadOutMapping::~DTReadOutMapping() {}

DTReadOutGeometryLink::~DTReadOutGeometryLink() {}

//--------------
// Operations --
//--------------
int DTReadOutMapping::readOutToGeometry(
    int dduId, int rosId, int robId, int tdcId, int channelId, DTWireId& wireId) const {
  int wheelId;
  int stationId;
  int sectorId;
  int slId;
  int layerId;
  int cellId;

  int status =
      readOutToGeometry(dduId, rosId, robId, tdcId, channelId, wheelId, stationId, sectorId, slId, layerId, cellId);

  wireId = DTWireId(wheelId, stationId, sectorId, slId, layerId, cellId);
  return status;
}

int DTReadOutMapping::readOutToGeometry(int dduId,
                                        int rosId,
                                        int robId,
                                        int tdcId,
                                        int channelId,
                                        int& wheelId,
                                        int& stationId,
                                        int& sectorId,
                                        int& slId,
                                        int& layerId,
                                        int& cellId) const {
  wheelId = stationId = sectorId = slId = layerId = cellId = 0;

  if (!atomicCache().isSet()) {
    cacheMap();
  }

  int defaultValue;
  atomicCache()->mType.find(0, defaultValue);
  if (defaultValue) {
    int searchStatus;
    int ientry;

    std::vector<int> dduKey;
    dduKey.reserve(5);
    dduKey.push_back(dduId);
    dduKey.push_back(rosId);
    searchStatus = atomicCache()->rgDDU.find(dduKey.begin(), dduKey.end(), ientry);
    if (searchStatus)
      return searchStatus;
    const DTReadOutGeometryLink& lros(readOutChannelDriftTubeMap[ientry]);
    wheelId = lros.wheelId;
    sectorId = lros.sectorId;

    std::vector<int> rosKey;
    rosKey.reserve(5);
    rosKey.push_back(lros.cellId);
    rosKey.push_back(robId);
    searchStatus = atomicCache()->rgROS.find(rosKey.begin(), rosKey.end(), ientry);
    if (searchStatus)
      return searchStatus;
    const DTReadOutGeometryLink& lrob(readOutChannelDriftTubeMap[ientry]);
    if (lrob.wheelId != defaultValue)
      wheelId = lrob.wheelId;
    stationId = lrob.stationId;
    if (lrob.sectorId != defaultValue)
      sectorId = lrob.sectorId;

    std::vector<int> robKey;
    robKey.reserve(5);
    robKey.push_back(lrob.cellId);
    robKey.push_back(tdcId);
    robKey.push_back(channelId);
    searchStatus = atomicCache()->rgROB.find(robKey.begin(), robKey.end(), ientry);
    if (searchStatus)
      return searchStatus;
    const DTReadOutGeometryLink& ltdc(readOutChannelDriftTubeMap[ientry]);
    slId = ltdc.slId;
    layerId = ltdc.layerId;
    cellId = ltdc.cellId;
    return 0;
  }

  std::vector<int> chanKey;
  chanKey.reserve(5);
  chanKey.push_back(dduId);
  chanKey.push_back(rosId);
  chanKey.push_back(robId);
  chanKey.push_back(tdcId);
  chanKey.push_back(channelId);
  int ientry;
  int searchStatus = atomicCache()->rgBuf.find(chanKey.begin(), chanKey.end(), ientry);
  if (!searchStatus) {
    const DTReadOutGeometryLink& link(readOutChannelDriftTubeMap[ientry]);
    wheelId = link.wheelId;
    stationId = link.stationId;
    sectorId = link.sectorId;
    slId = link.slId;
    layerId = link.layerId;
    cellId = link.cellId;
  }

  return searchStatus;
}

int DTReadOutMapping::geometryToReadOut(
    const DTWireId& wireId, int& dduId, int& rosId, int& robId, int& tdcId, int& channelId) const {
  return geometryToReadOut(wireId.wheel(),
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

int DTReadOutMapping::geometryToReadOut(int wheelId,
                                        int stationId,
                                        int sectorId,
                                        int slId,
                                        int layerId,
                                        int cellId,
                                        int& dduId,
                                        int& rosId,
                                        int& robId,
                                        int& tdcId,
                                        int& channelId) const {
  dduId = rosId = robId = tdcId = channelId = 0;

  if (!atomicCache().isSet()) {
    cacheMap();
  }

  int defaultValue;
  atomicCache()->mType.find(0, defaultValue);
  if (defaultValue) {
    int searchStatus;
    int mapId = 0;
    std::vector<int> const* robMLgr;
    std::vector<int> const* rosMLgr;
    std::vector<int> const* dduMLgr;

    std::vector<int> cellKey;
    cellKey.reserve(6);
    cellKey.push_back(cellId);
    cellKey.push_back(layerId);
    cellKey.push_back(slId);
    std::vector<int> stdcKey = cellKey;
    searchStatus = atomicCache()->grROB.find(cellKey.begin(), cellKey.end(), robMLgr);
    if (searchStatus)
      return searchStatus;
    if (robMLgr->empty())
      return 1;
    std::vector<int>::const_iterator tdc_iter = robMLgr->begin();
    std::vector<int>::const_iterator tdc_iend = robMLgr->end();
    while (tdc_iter != tdc_iend) {
      const DTReadOutGeometryLink& ltdc(readOutChannelDriftTubeMap[*tdc_iter++]);
      channelId = ltdc.channelId;
      tdcId = ltdc.tdcId;
      mapId = ltdc.rosId;
      cellKey.clear();
      cellKey.push_back(mapId);
      cellKey.push_back(stationId);
      std::vector<int> srosKey = cellKey;
      searchStatus = atomicCache()->grROS.find(cellKey.begin(), cellKey.end(), rosMLgr);
      if (searchStatus)
        continue;
      if (rosMLgr->empty())
        continue;
      std::vector<int>::const_iterator ros_iter = rosMLgr->begin();
      std::vector<int>::const_iterator ros_iend = rosMLgr->end();
      while (ros_iter != ros_iend) {
        const DTReadOutGeometryLink& lros(readOutChannelDriftTubeMap[*ros_iter++]);
        int secCk = lros.sectorId;
        int wheCk = lros.wheelId;
        if ((secCk != defaultValue) && (secCk != sectorId))
          continue;
        if ((wheCk != defaultValue) && (wheCk != wheelId))
          continue;
        robId = lros.robId;
        mapId = lros.rosId;
        cellKey.clear();
        cellKey.push_back(mapId);
        cellKey.push_back(wheelId);
        cellKey.push_back(sectorId);
        std::vector<int> sdduKey = cellKey;
        searchStatus = atomicCache()->grDDU.find(cellKey.begin(), cellKey.end(), dduMLgr);
        if (searchStatus)
          continue;
        if (dduMLgr->empty())
          continue;
        if (searchStatus)
          return searchStatus;
        if (dduMLgr->empty())
          return 1;
        std::vector<int>::const_iterator ddu_iter = dduMLgr->begin();
        std::vector<int>::const_iterator ddu_iend = dduMLgr->end();
        while (ddu_iter != ddu_iend) {
          const DTReadOutGeometryLink& lddu(readOutChannelDriftTubeMap[*ddu_iter++]);
          if (((sectorId == secCk) || (sectorId == lddu.sectorId)) &&
              ((wheelId == wheCk) || (wheelId == lddu.wheelId))) {
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
  cellKey.push_back(wheelId);
  cellKey.push_back(stationId);
  cellKey.push_back(sectorId);
  cellKey.push_back(slId);
  cellKey.push_back(layerId);
  cellKey.push_back(cellId);
  int ientry;
  int searchStatus = atomicCache()->grBuf.find(cellKey.begin(), cellKey.end(), ientry);
  if (!searchStatus) {
    const DTReadOutGeometryLink& link(readOutChannelDriftTubeMap[ientry]);
    dduId = link.dduId;
    rosId = link.rosId;
    robId = link.robId;
    tdcId = link.tdcId;
    channelId = link.channelId;
  }

  return searchStatus;
}

DTReadOutMapping::type DTReadOutMapping::mapType() const {
  if (!atomicCache().isSet()) {
    cacheMap();
  }

  int defaultValue;
  atomicCache()->mType.find(0, defaultValue);
  if (defaultValue)
    return compact;
  else
    return plain;
}

const std::string& DTReadOutMapping::mapCellTdc() const { return cellMapVersion; }

std::string& DTReadOutMapping::mapCellTdc() { return cellMapVersion; }

const std::string& DTReadOutMapping::mapRobRos() const { return robMapVersion; }

std::string& DTReadOutMapping::mapRobRos() { return robMapVersion; }

void DTReadOutMapping::clear() {
  atomicCache().reset();
  rgBuf->clear();
  grBuf->clear();
  readOutChannelDriftTubeMap.clear();
  return;
}

int DTReadOutMapping::insertReadOutGeometryLink(int dduId,
                                                int rosId,
                                                int robId,
                                                int tdcId,
                                                int channelId,
                                                int wheelId,
                                                int stationId,
                                                int sectorId,
                                                int slId,
                                                int layerId,
                                                int cellId) {
  DTReadOutGeometryLink link;
  link.dduId = dduId;
  link.rosId = rosId;
  link.robId = robId;
  link.tdcId = tdcId;
  link.channelId = channelId;
  link.wheelId = wheelId;
  link.stationId = stationId;
  link.sectorId = sectorId;
  link.slId = slId;
  link.layerId = layerId;
  link.cellId = cellId;

  int ientry = readOutChannelDriftTubeMap.size();
  readOutChannelDriftTubeMap.push_back(link);

  DTBufferTree<int, int>* pgrBuf;
  DTBufferTree<int, int>* prgBuf;

  if (atomicCache().isSet()) {
    pgrBuf = &atomicCache()->grBuf;
    prgBuf = &atomicCache()->rgBuf;
  } else {
    pgrBuf = grBuf.get();
    prgBuf = rgBuf.get();
  }

  std::vector<int> cellKey;
  cellKey.reserve(6);
  cellKey.push_back(wheelId);
  cellKey.push_back(stationId);
  cellKey.push_back(sectorId);
  cellKey.push_back(slId);
  cellKey.push_back(layerId);
  cellKey.push_back(cellId);
  int grStatus = pgrBuf->insert(cellKey.begin(), cellKey.end(), ientry);

  std::vector<int> chanKey;
  chanKey.reserve(5);
  chanKey.push_back(dduId);
  chanKey.push_back(rosId);
  chanKey.push_back(robId);
  chanKey.push_back(tdcId);
  chanKey.push_back(channelId);
  int rgStatus = prgBuf->insert(chanKey.begin(), chanKey.end(), ientry);

  if (grStatus || rgStatus)
    return 1;
  else
    return 0;
}

DTReadOutMapping::const_iterator DTReadOutMapping::begin() const { return readOutChannelDriftTubeMap.begin(); }

DTReadOutMapping::const_iterator DTReadOutMapping::end() const { return readOutChannelDriftTubeMap.end(); }

const DTReadOutMapping* DTReadOutMapping::fullMap() const {
  if (mapType() == plain)
    return this;
  return expandMap(*this);
}

// The code for this function was copied verbatim from
// CondCore/DTPlugins/src/DTCompactMapPluginHandler.c
DTReadOutMapping* DTReadOutMapping::expandMap(const DTReadOutMapping& compMap) {
  std::vector<DTReadOutGeometryLink> entryList;
  DTReadOutMapping::const_iterator compIter = compMap.begin();
  DTReadOutMapping::const_iterator compIend = compMap.end();
  while (compIter != compIend)
    entryList.push_back(*compIter++);

  std::string rosMap = "expand_";
  rosMap += compMap.mapRobRos();
  std::string tdcMap = "expand_";
  tdcMap += compMap.mapCellTdc();
  DTReadOutMapping* fullMap = new DTReadOutMapping(tdcMap, rosMap);
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
  while (iter != iend) {
    const DTReadOutGeometryLink& rosEntry(*iter++);
    if (rosEntry.dduId > 0x3fffffff)
      continue;
    ddu = rosEntry.dduId;
    ros = rosEntry.rosId;
    whe = rosEntry.wheelId;
    def = rosEntry.stationId;
    sec = rosEntry.sectorId;
    rob = rosEntry.slId;
    mt1 = rosEntry.layerId;
    mi1 = rosEntry.cellId;
    iros = entryList.begin();
    while (iros != iend) {
      wha = whe;
      sea = sec;
      const DTReadOutGeometryLink& rchEntry(*iros++);
      if ((rchEntry.dduId != mt1) || (rchEntry.rosId != mi1))
        continue;
      rch = rchEntry.robId;
      if (rchEntry.wheelId != def)
        wha = rchEntry.wheelId;
      sta = rchEntry.stationId;
      if (rchEntry.sectorId != def)
        sea = rchEntry.sectorId;
      rob = rchEntry.slId;
      mt2 = rchEntry.layerId;
      mi2 = rchEntry.cellId;
      irob = entryList.begin();
      while (irob != iend) {
        const DTReadOutGeometryLink& robEntry(*irob++);
        if ((robEntry.dduId != mt2) || (robEntry.rosId != mi2))
          continue;
        if (robEntry.robId != rob) {
          std::cout << "ROB mismatch " << rob << " " << robEntry.robId << std::endl;
        }
        tdc = robEntry.tdcId;
        tch = robEntry.channelId;
        qua = robEntry.slId;
        lay = robEntry.layerId;
        cel = robEntry.cellId;
        fullMap->insertReadOutGeometryLink(ddu, ros, rch, tdc, tch, wha, sta, sea, qua, lay, cel);
      }
    }
  }
  return fullMap;
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
  std::unique_ptr<DTReadOutMappingCache> localCache(new DTReadOutMappingCache);

  localCache->mType.insert(0, 0);

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
  std::map<int, std::vector<int>*> dduEntries;
  for (entryNum = 0; entryNum < entryMax; entryNum++) {
    const DTReadOutGeometryLink& link(readOutChannelDriftTubeMap[entryNum]);

    key = link.dduId;
    val = link.stationId;
    if (key > 0x3fffffff) {
      if (link.tdcId > 0x3fffffff) {
        localCache->mType.insert(0, defaultValue = link.tdcId);
        rosMapKey = key;
      } else {
        localCache->mType.insert(0, defaultValue = link.wheelId);
        robMapKey = key;
      }
    }

    if (defaultValue == 0) {
      chanKey.clear();
      chanKey.push_back(link.dduId);
      chanKey.push_back(link.rosId);
      chanKey.push_back(link.robId);
      chanKey.push_back(link.tdcId);
      chanKey.push_back(link.channelId);

      localCache->rgBuf.insert(chanKey.begin(), chanKey.end(), entryNum);

      cellKey.clear();
      cellKey.push_back(link.wheelId);
      cellKey.push_back(link.stationId);
      cellKey.push_back(link.sectorId);
      cellKey.push_back(link.slId);
      cellKey.push_back(link.layerId);
      cellKey.push_back(link.cellId);

      localCache->grBuf.insert(cellKey.begin(), cellKey.end(), entryNum);
    }

    if (key == robMapKey) {
      chanKey.clear();
      chanKey.push_back(link.rosId);
      chanKey.push_back(link.tdcId);
      chanKey.push_back(link.channelId);
      localCache->rgROB.insert(chanKey.begin(), chanKey.end(), entryNum);

      cellKey.clear();
      cellKey.push_back(link.cellId);
      cellKey.push_back(link.layerId);
      cellKey.push_back(link.slId);
      std::vector<int>* robMLgr;
      localCache->grROB.find(cellKey.begin(), cellKey.end(), robMLgr);
      if (robMLgr == nullptr) {
        std::unique_ptr<std::vector<int> > newVector(new std::vector<int>);
        robMLgr = newVector.get();
        localCache->grROB.insert(cellKey.begin(), cellKey.end(), std::move(newVector));
      }
      robMLgr->push_back(entryNum);
    }

    if (key == rosMapKey) {
      chanKey.clear();
      chanKey.push_back(link.rosId);
      chanKey.push_back(link.robId);
      localCache->rgROS.insert(chanKey.begin(), chanKey.end(), entryNum);

      cellKey.clear();
      cellKey.push_back(link.cellId);
      cellKey.push_back(link.stationId);
      std::vector<int>* rosMLgr;
      localCache->grROS.find(cellKey.begin(), cellKey.end(), rosMLgr);
      if (rosMLgr == nullptr) {
        std::unique_ptr<std::vector<int> > newVector(new std::vector<int>);
        rosMLgr = newVector.get();
        localCache->grROS.insert(cellKey.begin(), cellKey.end(), std::move(newVector));
      }
      rosMLgr->push_back(entryNum);
    }

    if ((key < 0x3fffffff) && (val > 0x3fffffff)) {
      chanKey.clear();
      chanKey.push_back(link.dduId);
      chanKey.push_back(link.rosId);
      localCache->rgDDU.insert(chanKey.begin(), chanKey.end(), entryNum);

      int mapId = link.cellId;
      std::vector<int>* dduMLgr;
      std::map<int, std::vector<int>*>::const_iterator dduEntIter = dduEntries.find(mapId);
      if (dduEntIter == dduEntries.end())
        dduEntries.insert(std::pair<int, std::vector<int>*>(mapId, dduMLgr = new std::vector<int>));
      else
        dduMLgr = dduEntIter->second;
      dduMLgr->push_back(entryNum);
    }
  }

  if (defaultValue != 0) {
    for (entryNum = 0; entryNum < entryMax; entryNum++) {
      const DTReadOutGeometryLink& link(readOutChannelDriftTubeMap[entryNum]);
      key = link.dduId;
      if (key != rosMapKey)
        continue;
      int mapId = link.rosId;
      int whchkId = link.wheelId;
      int secchkId = link.sectorId;

      std::vector<int>* dduMLgr;
      std::map<int, std::vector<int>*>::const_iterator dduEntIter = dduEntries.find(mapId);
      if (dduEntIter != dduEntries.end())
        dduMLgr = dduEntIter->second;
      else
        continue;
      std::vector<int>::const_iterator dduIter = dduMLgr->begin();
      std::vector<int>::const_iterator dduIend = dduMLgr->end();
      while (dduIter != dduIend) {
        int ientry = *dduIter++;
        const DTReadOutGeometryLink& lros(readOutChannelDriftTubeMap[ientry]);
        int wheelId = whchkId;
        int sectorId = secchkId;
        if (wheelId == defaultValue)
          wheelId = lros.wheelId;
        if (sectorId == defaultValue)
          sectorId = lros.sectorId;
        cellKey.clear();
        cellKey.push_back(mapId);
        cellKey.push_back(wheelId);
        cellKey.push_back(sectorId);
        std::vector<int>* dduMLgr = nullptr;
        localCache->grDDU.find(cellKey.begin(), cellKey.end(), dduMLgr);
        if (dduMLgr == nullptr) {
          std::unique_ptr<std::vector<int> > newVector(new std::vector<int>);
          dduMLgr = newVector.get();
          localCache->grDDU.insert(cellKey.begin(), cellKey.end(), std::move(newVector));
        }
        dduMLgr->push_back(ientry);
      }
    }

    std::map<int, std::vector<int>*>::const_iterator dduEntIter = dduEntries.begin();
    std::map<int, std::vector<int>*>::const_iterator dduEntIend = dduEntries.end();
    while (dduEntIter != dduEntIend) {
      const std::pair<int, std::vector<int>*>& dduEntry = *dduEntIter++;
      delete dduEntry.second;
    }

    localCache->rgBuf.clear();
    localCache->grBuf.clear();
  }

  atomicCache().set(std::move(localCache));

  return;
}
