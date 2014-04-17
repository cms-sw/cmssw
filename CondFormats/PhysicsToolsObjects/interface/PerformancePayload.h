#ifndef PerformancePayload_h
#define PerformancePayload_h

//#include "CondFormats/PerformanceDBObjects/interface/PhysicsPerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/BinningPointByMap.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceResult.h"


#include <string>
#include <vector>
#include <iostream>

class PerformancePayload
// : public PhysicsPerformancePayload 
{
 public:

  static const float InvalidResult;

  //    PerformancePayload(int stride_, std::string columns_,std::vector<float> table) : PhysicsPerformancePayload(stride_, columns_, table) {}

  PerformancePayload(){}
  virtual ~PerformancePayload() {};

  virtual void initialize() { }

  virtual float getResult(PerformanceResult::ResultType,const BinningPointByMap&) const = 0; // gets from the full payload
  virtual bool isInPayload(PerformanceResult::ResultType,const BinningPointByMap&) const = 0;
 protected:

};

#endif

