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
  bool isGood(int mId) const { return ! this->isBad(mId); };

  double responseFraction = 1.0;
  void generateResponseFraction(const std::map<int, float>& fractionalResponses, const std::vector<int>& baseVeto = std::vector<int>()){
    usingBaseVeto = ! baseVeto.empty();
    double responseTotal = 0;
    responseFraction = 0;
    for (const auto& [modID, frac] : fractionalResponses) {
      responseTotal += frac;
      if (this->isBad(modID)) continue;
      if (std::find(baseVeto.begin(), baseVeto.end(), modID) != baseVeto.end()) continue;
      responseFraction += frac;
    }
    responseFraction /= responseTotal;
  }

private:
  std::vector<int> badModules;
  bool usingBaseVeto = false;
  // double responseFraction = 1.0; // is public not to have to use a getter
  
  COND_SERIALIZABLE;
};


#endif
