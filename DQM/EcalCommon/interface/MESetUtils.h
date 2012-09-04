#ifndef MESetUtils_H
#define MESetUtils_H

#include "DQM/EcalCommon/interface/MESet.h"

namespace edm
{
  class ParameterSet;
}

namespace ecaldqm
{
  MESet* createMESet(edm::ParameterSet const&, BinService const*, std::string = "", DQMStore* = 0);
}

#endif
