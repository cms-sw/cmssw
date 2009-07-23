#ifndef BtagPerformancePayload_h
#define BtagPerformancePayload_h

#include "CondFormats/BTauObjects/interface/PhysicsPerformancePayload.h"
#include "CondFormats/BTauObjects/interface/BtagBinningPointByMap.h"
#include "CondFormats/BTauObjects/interface/BtagResult.h"


#include <string>
#include <vector>

class BtagPerformancePayload : public PhysicsPerformancePayload {
 public:

  static const float InvalidResult;

  BtagPerformancePayload(int stride_, std::string columns_,std::vector<float> table) : PhysicsPerformancePayload(stride_, columns_, table) {}

  BtagPerformancePayload(){}

  virtual float getResult(BtagResult::BtagResultType,BtagBinningPointByMap) const = 0; // gets from the full payload
  virtual bool isInPayload(BtagResult::BtagResultType,BtagBinningPointByMap) const = 0;
 protected:

};

#endif

