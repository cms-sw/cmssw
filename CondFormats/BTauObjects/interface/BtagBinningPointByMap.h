#ifndef BtagBinningPointByMap_h
#define BtagBinningPointByMap_h


#include <map>

class BtagBinningPointByMap {
 public:
  enum  BtagBinningPointType{Eta=1, JetEt=2, Phi=3, NTracks=4};

  typedef std::map<BtagBinningPointType, float> BtagBinningPointTypeMap;

  bool insert(BtagBinningPointType, float);

  float value(BtagBinningPointType);

  bool isKeyAvailable(BtagBinningPointType);

  void reset() {map_.clear();}

  const BtagBinningPointTypeMap & map(){return map_;}
  

 private:
  BtagBinningPointTypeMap map_;
};


#endif
