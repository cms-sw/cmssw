#ifndef BtagPerformancePayloadFromTableEtaJetEtPhi_h
#define BtagPerformancePayloadFromTableEtaJetEtPhi_h

#include "CondFormats/BTauObjects/interface/BtagPerformancePayloadFromTable.h"


#include <string>
#include <vector>

class BtagPerformancePayloadFromTableEtaJetEtPhi : public BtagPerformancePayloadFromTable {
 public:

  BtagPerformancePayloadFromTableEtaJetEtPhi(int stride_, std::string columns_,std::vector<float> table);

  BtagPerformancePayloadFromTableEtaJetEtPhi(){}

 protected:


  virtual std::vector<BtagBinningPointByMap::BtagBinningPointType> myBinning() const {
    std::vector<BtagBinningPointByMap::BtagBinningPointType> temp;
    temp.push_back(BtagBinningPointByMap::Eta);
    temp.push_back(BtagBinningPointByMap::JetEt);
    temp.push_back(BtagBinningPointByMap::Phi);
    return temp;
  }

  
  virtual int minPos(const BtagBinningPointByMap::BtagBinningPointType t) const {
    switch (t){
    case BtagBinningPointByMap::Eta:
      return 0;
      break;
    case BtagBinningPointByMap::JetEt:
      return 2;
      break;
    case BtagBinningPointByMap::Phi:
      return 4;
      break;
    default:
      return BtagPerformancePayloadFromTable::InvalidPos;
      break; 
    }
  }
  virtual int maxPos(const BtagBinningPointByMap::BtagBinningPointType t) const {
    switch (t){
    case BtagBinningPointByMap::Eta:
      return 1;
      break;
    case BtagBinningPointByMap::JetEt:
      return 3;
      break;
    case BtagBinningPointByMap::Phi:
      return 5;
      break;
    default:
      return BtagPerformancePayloadFromTable::InvalidPos;
      break; 
    }
  }

  virtual int resultPos(BtagResult::BtagResultType r) const {
    switch (r){
    case BtagResult::BEFF: 
      return 6;
      break;
    case BtagResult::BERR: 
      return 7;
      break;
    case BtagResult::CEFF: 
      return 8;
      break;
    case BtagResult::CERR: 
      return 9;
      break;
    case BtagResult::LEFF: 
      return 10;
      break;
    case BtagResult::LERR: 
      return 11;
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

