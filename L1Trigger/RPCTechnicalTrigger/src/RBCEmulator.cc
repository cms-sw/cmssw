#include "L1Trigger/RBCEmulator/interface/RBCEmulator.h"
#include "L1Trigger/RBCEmulator/interface/RBCOutputSignalContainer.h"
#include "L1Trigger/RBCEmulator/interface/RBCOutputSignal.h"
#include "L1Trigger/RBCEmulator/interface/RBCPolicy.h"
#include "L1Trigger/RBCEmulator/src/RBCLogic.h"
#include "L1Trigger/RBCEmulator/src/RBCPatternLogic.h"
#include "L1Trigger/RBCEmulator/src/RBCChamberORLogic.h"


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


