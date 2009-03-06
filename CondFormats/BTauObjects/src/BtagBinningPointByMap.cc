#include "CondFormats/BTauObjects/interface/BtagBinningPointByMap.h"

bool BtagBinningPointByMap::insert(BtagBinningPointType k, float v){
  map_[k]=v;
  return true;
}
bool BtagBinningPointByMap::isKeyAvailable(BtagBinningPointType k){
  return (map_.find(k) != map_.end());
}

float BtagBinningPointByMap::value(BtagBinningPointType k){
  if (isKeyAvailable(k) == false) return -100;
  return map_[k];

}

