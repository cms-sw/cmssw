#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/FebLocationSpec.h"
#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"

#include "CondFormats/RPCObjects/interface/FebConnectorSpec.h"
#include "CondFormats/RPCObjects/interface/L1RPCConfig.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

#include "CondFormats/RPCObjects/interface/RPCdbData.h"

#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"

namespace{
  std::vector<ChamberStripSpec> theStrips;
}


#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
namespace{
  std::vector<FebConnectorSpec> theFebs;
}

#include "CondFormats/RPCObjects/interface/LinkConnSpec.h"
namespace{
  std::vector<LinkBoardSpec> theLBs; 
}

#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
namespace{
  std::vector<LinkConnSpec> theLinks; 
}

#include "CondFormats/RPCObjects/interface/DccSpec.h"
namespace{
  std::vector<TriggerBoardSpec> theTBs; 
}

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
namespace{
  std::map<int, DccSpec> theFeds; 
}

#include "CondFormats/RPCObjects/interface/RPCEMap.h"



