#ifndef BtagPerformance_h
#define BtagPerformance_h

//#include "CondFormats/BTagPerformance/interface/BtagPerformanceInterface.h"

#include "CondFormats/BTauObjects/interface/BtagPerformancePayload.h"
#include "CondFormats/BTauObjects/interface/BtagWorkingPoint.h"


#include <string>
#include <vector>

class BtagPerformance {
public:
  BtagPerformance(const BtagPerformancePayload& p, const BtagWorkingPoint& w) : pl(p), wp(w) {}

  virtual float getResult(BtagResult::BtagResultType, BtagBinningPointByMap) const ;

  virtual bool isResultOk(BtagResult::BtagResultType, BtagBinningPointByMap) const ;
  
  virtual const BtagWorkingPoint& workingPoint() const {return wp;}

  private:
  const BtagPerformancePayload& pl;
  const BtagWorkingPoint& wp;

};


#endif

