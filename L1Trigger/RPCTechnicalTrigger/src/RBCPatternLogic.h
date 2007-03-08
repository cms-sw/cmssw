#ifndef RPCTechnicalTrigger_RBCPatternLogic_h
#define RPCTechnicalTrigger_RBCPatternLogic_h
#include "L1Trigger/RPCTechnicalTrigger/src/RBCLogic.h"
#include <iostream>

class RBCPatternLogic : public RBCLogic
{
 public:
  RBCPatternLogic(){std::cout <<"RBCPatternLogic"<<std::endl;}
  virtual ~RBCPatternLogic(){ std::cout <<"bye Pattern Logic"<<std::endl;}
  void action(){std::cout <<" Pretending to do Pattern Logic"<<std::endl;}
  
};
#endif
