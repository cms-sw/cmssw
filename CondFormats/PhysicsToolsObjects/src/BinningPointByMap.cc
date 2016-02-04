#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"

bool BinningPointByMap::insert(BinningVariables::BinningVariablesType k, float v){
  map_[k]=v;
  return true;
}
bool BinningPointByMap::isKeyAvailable(BinningVariables::BinningVariablesType k){
  return (map_.find(k) != map_.end());
}

float BinningPointByMap::value(BinningVariables::BinningVariablesType k){
  if (isKeyAvailable(k) == false) return -100;
  return map_[k];

}

