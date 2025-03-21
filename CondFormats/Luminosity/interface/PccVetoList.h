#ifndef CondFormats_Luminosity_PccVetoList_h
#define CondFormats_Luminosity_PccVetoList_h

/** 
 * \class PccVetoList
 * 
 * \author Peter Major
 *  
 */

#include <sstream>
#include <cstring>
#include <algorithm>
#include <vector>
#include <boost/serialization/vector.hpp>
#include "CondFormats/Serialization/interface/Serializable.h"

class PccVetoList {
public:
  void addToVetoList( const std::vector<int>& modList ){ 
    badModules.reserve(badModules.size()+modList.size()); 
    badModules.insert(badModules.end(), modList.begin(), modList.end()); 
  }
  const std::vector<int>& getBadModules() const { return badModules; }
  bool isBad(int mId)  const { return (std::find(badModules.begin(), badModules.end(), mId) != badModules.end()); };
  bool isGood(int mId) const { return (std::find(badModules.begin(), badModules.end(), mId) == badModules.end()); };

private:
  std::vector<int> badModules;
  COND_SERIALIZABLE;
};
#endif
