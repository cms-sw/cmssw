#include "L1Trigger/RPCTechnicalTrigger/interface/RBCOutputSignalContainer.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCOutputSignal.h"


RBCOutputSignalContainer::
RBCOutputSignalContainer(){}


RBCOutputSignalContainer::
~RBCOutputSignalContainer(){}


void 
RBCOutputSignalContainer::
insert(RBCOutputSignal s)
{
  signals.insert(s);
}

RBCOutputSignalContainer::
iterator
RBCOutputSignalContainer::
begin()
{
  return signals.begin();
}

RBCOutputSignalContainer::
iterator
RBCOutputSignalContainer::
end()
{
  return signals.end();
}
