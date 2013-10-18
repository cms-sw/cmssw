#include "CondFormats/RPCObjects/src/headers.h"


namespace{
  struct dictionary {
    std::vector<ChamberStripSpec> theStrips;
 
    std::vector<FebConnectorSpec> theFebs;
 
    std::vector<LinkBoardSpec> theLBs; 
 
    std::vector<LinkConnSpec> theLinks; 
 
    std::vector<TriggerBoardSpec> theTBs; 
 
    std::map<int, DccSpec> theFeds; 
  };
}

