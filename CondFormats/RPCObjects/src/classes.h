#include "CondFormats/RPCObjects/interface/ChamberStripSpec.h"
#include "CondFormats/RPCObjects/interface/FebSpec.h"
namespace{
  std::vector<ChamberStripSpec> theStrips;
}

#include "CondFormats/RPCObjects/interface/ChamberLocationSpec.h"

#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
namespace{
  std::vector<FebSpec> theFebs;
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





