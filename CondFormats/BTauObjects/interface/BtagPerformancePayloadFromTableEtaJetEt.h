#ifndef BtagPerformancePayloadFromTableEtaJetEt_h
#define BtagPerformancePayloadFromTableEtaJetEt_h

#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTable.h"
#include "CondFormats/BTauObjects/interface/BtagBinningPointByMap.h"


#include <string>
#include <vector>

class BtagPerformancePayloadFromTableEtaJetEt : public BtagPerformancePayloadFromTable {
 public:
  BtagPerformancePayloadFromTableEtaJetEt(int stride_, std::string columns_,std::vector<float> table);
  BtagPerformancePayloadFromTableEtaJetEt(){}
 protected:


  virtual std::vector<BtagBinningPointByMap::BtagBinningPointType> myBinning() const {
    std::vector<BtagBinningPointByMap::BtagBinningPointType> temp;
    temp.push_back(BtagBinningPointByMap::Eta);
    temp.push_back(BtagBinningPointByMap::JetEt);
    return temp;
  }


  
  virtual int minPos(BtagBinningPointByMap::BtagBinningPointType t) const {
    switch (t){
    case BtagBinningPointByMap::Eta:
      return 0;
      break;
    case BtagBinningPointByMap::JetEt:
      return 2;
      break;
    default:
      return BtagPerformancePayloadFromTable::InvalidPos;
      break; 
    }
  };
  
  virtual int maxPos(BtagBinningPointByMap::BtagBinningPointType t) const {
    switch (t){
    case BtagBinningPointByMap::Eta:
      return 1;
      break;
    case BtagBinningPointByMap::JetEt:
      return 3;
      break;
    default:
      return BtagPerformancePayloadFromTable::InvalidPos;
      break; 
    }
  };
  
  
  virtual int resultPos(BtagResult::BtagResultType r) const {
    switch (r){
    case BtagResult::BEFF: 
      return 4;
      break;
    case BtagResult::BERR: 
      return 5;
      break;
    case BtagResult::CEFF: 
      return 6;
      break;
    case BtagResult::CERR: 
      return 7;
      break;
    case BtagResult::LEFF: 
      return 8;
      break;
    case BtagResult::LERR: 
      return 9;
      break;
    case BtagResult::NBEFF: 
      return BtagPerformancePayloadFromTable::InvalidPos;
      break;
    case BtagResult::NBERR: 
      return BtagPerformancePayloadFromTable::InvalidPos;
      break;
    default:
      return BtagPerformancePayloadFromTable::InvalidPos;
      break;
    }
  }
};

#endif

