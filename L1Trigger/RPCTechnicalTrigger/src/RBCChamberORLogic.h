#ifndef RBCEmulator_RBCChamberORLogic_h
#define RBCEmulator_RBCChamberORLogic_h
#include "L1Trigger/RBCTechnicalTrigger/src/RBCLogic.h"
#include <iostream>

class RBCChamberORLogic : public RBCLogic{
 public:
  RBCChamberORLogic(){std::cout <<"RBCChamberORLogic"<<std::endl;}
  virtual ~RBCChamberORLogic(){ std::cout <<"bye ChamberOR Logic"<<std::endl;}
  void action(){std::cout <<" Pretending to do ChamberOR Logic"<<std::endl;}
  
};

#endif
