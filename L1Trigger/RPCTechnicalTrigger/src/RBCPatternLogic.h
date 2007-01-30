#ifndef RBCEmulator_RBCPatternLogic_h
#define RBCEmulator_RBCPatternLogic_h
#include "L1Trigger/RBCEmulator/src/RBCLogic.h"
#include <iostream>

class RBCPatternLogic : public RBCLogic
{
 public:
  RBCPatternLogic(){std::cout <<"RBCPatternLogic"<<std::endl;}
  virtual ~RBCPatternLogic(){ std::cout <<"bye Pattern Logic"<<std::endl;}
  void action(){std::cout <<" Pretending to do Pattern Logic"<<std::endl;}
  
};

#endif

