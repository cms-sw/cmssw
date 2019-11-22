#ifndef CondTools_DT_DTKeyedConfigCache_h
#define CondTools_DT_DTKeyedConfigCache_h
/** \class DTKeyedConfigCache
 *
 *  Description:
 *       Class to hold configuration identifier for chambers
 *
 *  This class was originally defined in
 *  CondCore/DTPlugins/interface/DTConfigPluginHandler.h
 *  It was moved, renamed, and modified to not be a singleton
 *  for thread safety, but otherwise little was changed.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

#include <map>
#include <string>
#include <vector>

#include "CondCore/CondDB/interface/KeyList.h"

class DTKeyedConfig;

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTKeyedConfigCache {
public:
  DTKeyedConfigCache();
  virtual ~DTKeyedConfigCache();

  int get(const cond::persistency::KeyList& keyList, int cfgId, const DTKeyedConfig*& obj);

  void getData(const cond::persistency::KeyList& keyList, int cfgId, std::vector<std::string>& list);

  void purge();

  static const int maxBrickNumber;
  static const int maxStringNumber;
  static const int maxByteNumber;

private:
  DTKeyedConfigCache(const DTKeyedConfigCache& x) = delete;
  const DTKeyedConfigCache& operator=(const DTKeyedConfigCache& x) = delete;

  typedef std::pair<int, const DTKeyedConfig*> counted_brick;
  std::map<int, counted_brick> brickMap;
  int cachedBrickNumber;
  int cachedStringNumber;
  int cachedByteNumber;
};
#endif
