#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/HIObjects/interface/RPFlatParams.h"

#include <vector>

namespace {  
  struct dictionary{
    std::vector<CentralityTable::CBin> dummy;
    std::vector<RPFlatParams::EP> yummy;
  };
}

