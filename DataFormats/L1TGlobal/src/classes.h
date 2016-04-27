#include <vector>
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapRecord.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"
#include "DataFormats/L1TGlobal/interface/GlobalObject.h"

namespace DataFormats_L1TGlobal {
  struct dictionary {

    GlobalAlgBlkBxCollection                     dummy1a;
    GlobalExtBlkBxCollection                     dummy1b;    
    edm::Wrapper<GlobalAlgBlkBxCollection>       dummy1c;
    edm::Wrapper<GlobalExtBlkBxCollection>       dummy1d;
    
    GlobalObjectMap                              dummy2a;
    edm::Wrapper<GlobalObjectMap>                dummy2b;
    std::vector<GlobalObjectMap>                 dummy2c;
    edm::Wrapper<std::vector<GlobalObjectMap> >  dummy2d;

    GlobalObjectMapRecord                        dummy3a;
    edm::Wrapper<GlobalObjectMapRecord>          dummy3b; 

    std::vector<l1t::GlobalObject>               dummy4a;
    std::vector<std::vector<l1t::GlobalObject> > dummy4b;

    GlobalLogicParser::OperandToken              dummy5a;
    std::vector<GlobalLogicParser::OperandToken> dummy5b;

    GlobalLogicParser::TokenRPN                  dummy6a;
    std::vector<GlobalLogicParser::TokenRPN>     dummy6b;

    edm::Wrapper<GlobalAlgBlkBxCollection>    w_uGtAlgBxColl;
    edm::Wrapper<GlobalExtBlkBxCollection>    w_uGtExtBxColl;

    std::vector<GlobalAlgBlk> v_uGtAlgBx;
    std::vector<GlobalExtBlk> v_uGtExtBx;
  };
}
