#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"

using namespace egHLT;

void OffEvt::clear()
{
  jets_.clear();
  eles_.clear();
  phos_.clear();
  evtTrigBits_ = 0x0;
}
