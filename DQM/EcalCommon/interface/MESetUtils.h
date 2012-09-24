#ifndef MESetUtils_H
#define MESetUtils_H

#include "DQM/EcalCommon/interface/MESet.h"

namespace edm
{
  class ParameterSet;
}

namespace ecaldqm
{
  MESet* createMESet(edm::ParameterSet const&);

  void formPath(std::string&, std::map<std::string, std::string> const&);
}

#endif
