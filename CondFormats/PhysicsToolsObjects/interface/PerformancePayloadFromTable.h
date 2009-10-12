#ifndef PerformancePayloadFromTable_h
#define PerformancePayloadFromTable_h

#include "CondFormats/PhysicsToolsObjects/interface/PhysicsPerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"


#include <string>
#include <vector>

#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"

class PerformancePayloadFromTable : public PerformancePayload {
//  class PerformancePayloadFromTable : public PerformancePayload, public PhysicsPerformancePayload {
 public:

  static int InvalidPos;

  //PerformancePayloadFromTable(int stride_, std::string columns_,std::vector<float> table) : PerformancePayload(stride_, columns_, table) {}

    PerformancePayloadFromTable(std::vector<PerformanceResult::ResultType> r, std::vector<BinningVariables::BinningVariablesType> b , int stride_,std::vector<float> table) : 
      pl(stride_, table),
      results_(r), binning_(b) {}

  PerformancePayloadFromTable(){}
virtual ~PerformancePayloadFromTable(){}

  float getResult(PerformanceResult::ResultType,BinningPointByMap) const ; // gets from the full payload

  virtual bool isParametrizedInVariable(const BinningVariables::BinningVariablesType p)  const {
    return (minPos(p) != PerformancePayloadFromTable::InvalidPos);
  }
  
  virtual bool isInPayload(PerformanceResult::ResultType,BinningPointByMap) const ;

const PhysicsPerformancePayload & payLoad() const {return pl;}

 protected:

 virtual std::vector<BinningVariables::BinningVariablesType> myBinning() const {return binning_;}

  virtual int minPos(const BinningVariables::BinningVariablesType b) const {
    std::vector<BinningVariables::BinningVariablesType>::const_iterator p;
    p = find(binning_.begin(), binning_.end(), b);
    if (p == binning_.end()) return PerformancePayloadFromTable::InvalidPos;
    return ((p-binning_.begin())*2);
  }
  virtual int maxPos(const BinningVariables::BinningVariablesType b) const {
    std::vector<BinningVariables::BinningVariablesType>::const_iterator p;
    p = find(binning_.begin(), binning_.end(), b);
    if (p == binning_.end()) return PerformancePayloadFromTable::InvalidPos;
    return ((p-binning_.begin())*2+1);

  }    
    virtual int resultPos(PerformanceResult::ResultType r ) const {
      // result should be: # of binning variables *2 + position of result
      std::vector<PerformanceResult::ResultType>::const_iterator p;
      p = find (results_.begin(), results_.end(), r);
      if ( p == results_.end()) return PerformancePayloadFromTable::InvalidPos;
      return (binning_.size()*2+(p-results_.begin()));
      
  }

  bool matches(BinningPointByMap, PhysicsPerformancePayload::Row &) const;

  PhysicsPerformancePayload pl;

  std::vector<PerformanceResult::ResultType> results_;
  std::vector<BinningVariables::BinningVariablesType> binning_;
  
};

#endif

