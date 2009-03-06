#ifndef BtagPerformancePayloadFromTFormula_h
#define BtagPerformancePayloadFromTFormula_h

#include "CondFormats/BTauObjects/interface/BtagPerformancePayload.h"


#include <string>
#include <vector>

class BtagPerformancePayloadFromTFormula : public BtagPerformancePayload {
 public:

  static int InvalidPos;
  //
  // enforcement will be there
  BtagPerformancePayloadFromTFormula(int stride_, std::string columns_,std::vector<float> table) : BtagPerformancePayload(stride_, columns_, table) {}

  float getResult(BtagResult::BtagResultType,BtagBinningPoint) const ; // gets from the full payload

  virtual bool isInPayload(BtagResult::BtagResultType,BtagBinningPoint) const ;
 protected:
};

#endif

