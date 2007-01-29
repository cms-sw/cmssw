#include "L1Trigger/RPCTechnicalTrigger/interface/RBCEmulator.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCOutputSignalContainer.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCOutputSignal.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCPolicy.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCPatternLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/src/RBCChamberORLogic.h"


RBCEmulator::
RBCEmulator() : l(0)
{}

RBCEmulator::
~RBCEmulator()
{
}

void
RBCEmulator::
emulate(RBCPolicy* policy)
{
  l=policy->instance();
  std::cout <<"Setting the logic "<<policy->message()<<std::endl;
}

RBCOutputSignalContainer
RBCEmulator::triggers()
{
  RBCOutputSignalContainer c;
  l->action();
  return c;
}


