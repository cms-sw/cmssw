#ifndef BtagPerformancePayloadFromTable_h
#define BtagPerformancePayloadFromTable_h



#include "CondFormats/BTauObjects/interface/BtagPerformancePayload.h"


#include <string>
#include <vector>

#include "CondFormats/BTauObjects/interface/BtagBinningPointByMap.h"

class BtagPerformancePayloadFromTable : public BtagPerformancePayload {
 public:

  static int InvalidPos;

  BtagPerformancePayloadFromTable(int stride_, std::string columns_,std::vector<float> table) : BtagPerformancePayload(stride_, columns_, table) {}
  BtagPerformancePayloadFromTable(){}
  float getResult(BtagResult::BtagResultType,BtagBinningPointByMap) const ; // gets from the full payload

  virtual bool isParametrizedInVariable(const BtagBinningPointByMap::BtagBinningPointType p)  const {
    return (minPos(p) != BtagPerformancePayloadFromTable::InvalidPos);
  }
  
  virtual bool isInPayload(BtagResult::BtagResultType,BtagBinningPointByMap) const ;

 protected:

  virtual std::vector<BtagBinningPointByMap::BtagBinningPointType> myBinning() const =0;

  virtual int minPos(const BtagBinningPointByMap::BtagBinningPointType) const {return BtagPerformancePayloadFromTable::InvalidPos;}
  virtual int maxPos(const BtagBinningPointByMap::BtagBinningPointType) const {return BtagPerformancePayloadFromTable::InvalidPos;}
  virtual int resultPos(BtagResult::BtagResultType) const = 0;

  bool matches(BtagBinningPointByMap, PhysicsPerformancePayload::Row &) const;
  
};

#endif

