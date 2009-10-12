#ifndef BinningPointByMap_h
#define BinningPointByMap_h


#include "CondFormats/PhysicsToolsObjects/interface/BinningVariables.h"

#include <map>

class BinningPointByMap {
 public:
  //  enum  BinningPointType{Eta=1, JetEt=2, Phi=3, NTracks=4};

  typedef std::map<BinningVariables::BinningVariablesType, float> BinningPointTypeMap;

  bool insert(BinningVariables::BinningVariablesType, float);

  float value(BinningVariables::BinningVariablesType);

  bool isKeyAvailable(BinningVariables::BinningVariablesType);

  void reset() {map_.clear();}

  const BinningPointTypeMap & map(){return map_;}
  

 private:
  BinningPointTypeMap map_;
};


#endif
