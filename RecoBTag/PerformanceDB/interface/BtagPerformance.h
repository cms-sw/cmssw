#ifndef BtagPerformance_h
#define BtagPerformance_h

//#include "CondFormats/BTagPerformance/interface/BtagPerformanceInterface.h"

#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayload.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformanceWorkingPoint.h"


#include <string>
#include <vector>

class BtagPerformance {
public:
  BtagPerformance(const PerformancePayload& p, const PerformanceWorkingPoint& w) : pl(p), wp(w) {}

  virtual float getResult(PerformanceResult::ResultType, BinningPointByMap) const ;

  virtual bool isResultOk(PerformanceResult::ResultType, BinningPointByMap) const ;

  virtual const PerformancePayload & payload() const { return pl; }
  
  virtual const PerformanceWorkingPoint& workingPoint() const {return wp;}

  virtual ~BtagPerformance() {};

  private:
  const PerformancePayload& pl;
  const PerformanceWorkingPoint& wp;

};


#endif

