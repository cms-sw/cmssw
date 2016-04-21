#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapRecord.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"



namespace DataFormats_L1TGlobal {
  struct dictionary {

    GlobalAlgBlkBxCollection    uGtAlgBxColl;
    GlobalExtBlkBxCollection    uGtExtBxColl;    

    edm::Wrapper<GlobalAlgBlkBxCollection>    w_uGtAlgBxColl;
    edm::Wrapper<GlobalExtBlkBxCollection>    w_uGtExtBxColl;
    
    GlobalObjectMap  uGtObjectMap;
    edm::Wrapper<GlobalObjectMap> w_uGtObjectMap;

    GlobalObjectMapRecord  uGtObjectMapRecord;
    edm::Wrapper<GlobalObjectMapRecord> w_uGtObjectMapRecord;    
    
    
  };
}
