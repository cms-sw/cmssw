/*
 *  See header file for a description of this class.
 *
 *  This class was originally defined in
 *  CondCore/DTPlugins/src/DTConfigPluginHandler.cc
 *  It was moved, renamed, and modified to not be a singleton
 *  for thread safety, but otherwise little was changed.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTKeyedConfigCache.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTKeyedConfig.h"
#include "CondCore/CondDB/interface/KeyList.h"

#include <memory>
//-------------------
// Initializations --
//-------------------
const int DTKeyedConfigCache::maxBrickNumber = 5000;
const int DTKeyedConfigCache::maxStringNumber = 100000;
const int DTKeyedConfigCache::maxByteNumber = 10000000;

//----------------
// Constructors --
//----------------
DTKeyedConfigCache::DTKeyedConfigCache() : cachedBrickNumber(0), cachedStringNumber(0), cachedByteNumber(0) {}

//--------------
// Destructor --
//--------------
DTKeyedConfigCache::~DTKeyedConfigCache() { purge(); }

int DTKeyedConfigCache::get(const cond::persistency::KeyList& keyList, int cfgId, const DTKeyedConfig*& obj) {
  bool cacheFound = false;
  int cacheAge = 999999999;
  std::map<int, counted_brick>::iterator cache_iter = brickMap.begin();
  std::map<int, counted_brick>::iterator cache_icfg = brickMap.find(cfgId);
  std::map<int, counted_brick>::iterator cache_iend = brickMap.end();
  if (cache_icfg != cache_iend) {
    std::pair<const int, counted_brick>& entry = *cache_icfg;
    counted_brick& cBrick = entry.second;
    cacheAge = cBrick.first;
    obj = cBrick.second;
    cacheFound = true;
  }

  std::map<int, const DTKeyedConfig*> ageMap;
  if (cacheFound) {
    if (!cacheAge)
      return 0;
    while (cache_iter != cache_iend) {
      std::pair<const int, counted_brick>& entry = *cache_iter++;
      counted_brick& cBrick = entry.second;
      int& brickAge = cBrick.first;
      if (brickAge < cacheAge)
        brickAge++;
      if (entry.first == cfgId)
        brickAge = 0;
    }
    return 0;
  } else {
    while (cache_iter != cache_iend) {
      std::pair<const int, counted_brick>& entry = *cache_iter++;
      counted_brick& cBrick = entry.second;
      ageMap.insert(std::pair<int, const DTKeyedConfig*>(++cBrick.first, entry.second.second));
    }
  }

  std::shared_ptr<DTKeyedConfig> kBrick;
  bool brickFound = false;
  try {
    kBrick = keyList.getUsingKey<DTKeyedConfig>(cfgId);
    if (kBrick.get())
      brickFound = (kBrick->getId() == cfgId);
  } catch (std::exception const& e) {
  }
  if (brickFound) {
    counted_brick cBrick(0, obj = new DTKeyedConfig(*kBrick));
    brickMap.insert(std::pair<int, counted_brick>(cfgId, cBrick));
    DTKeyedConfig::data_iterator d_iter = kBrick->dataBegin();
    DTKeyedConfig::data_iterator d_iend = kBrick->dataEnd();
    cachedBrickNumber++;
    cachedStringNumber += (d_iend - d_iter);
    while (d_iter != d_iend)
      cachedByteNumber += (*d_iter++).size();
  }
  std::map<int, const DTKeyedConfig*>::reverse_iterator iter = ageMap.rbegin();
  while ((cachedBrickNumber > maxBrickNumber) || (cachedStringNumber > maxStringNumber) ||
         (cachedByteNumber > maxByteNumber)) {
    const DTKeyedConfig* oldestBrick = iter->second;
    int oldestId = oldestBrick->getId();
    cachedBrickNumber--;
    DTKeyedConfig::data_iterator d_iter = oldestBrick->dataBegin();
    DTKeyedConfig::data_iterator d_iend = oldestBrick->dataEnd();
    cachedStringNumber -= (d_iend - d_iter);
    while (d_iter != d_iend)
      cachedByteNumber -= (*d_iter++).size();
    brickMap.erase(oldestId);
    delete iter->second;
    iter++;
  }

  return 999;
}

void DTKeyedConfigCache::getData(const cond::persistency::KeyList& keyList, int cfgId, std::vector<std::string>& list) {
  const DTKeyedConfig* obj = nullptr;
  get(keyList, cfgId, obj);
  if (obj == nullptr)
    return;
  DTKeyedConfig::data_iterator d_iter = obj->dataBegin();
  DTKeyedConfig::data_iterator d_iend = obj->dataEnd();
  while (d_iter != d_iend)
    list.push_back(*d_iter++);
  DTKeyedConfig::link_iterator l_iter = obj->linkBegin();
  DTKeyedConfig::link_iterator l_iend = obj->linkEnd();
  while (l_iter != l_iend)
    getData(keyList, *l_iter++, list);
  return;
}

void DTKeyedConfigCache::purge() {
  std::map<int, counted_brick>::const_iterator iter = brickMap.begin();
  std::map<int, counted_brick>::const_iterator iend = brickMap.end();
  while (iter != iend) {
    delete iter->second.second;
    iter++;
  }
  brickMap.clear();
  cachedBrickNumber = 0;
  cachedStringNumber = 0;
  cachedByteNumber = 0;
  return;
}
