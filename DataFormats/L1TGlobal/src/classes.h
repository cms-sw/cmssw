#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "DataFormats/L1TGlobal/interface/L1TGtObjectMapRecord.h"
#include "DataFormats/L1TGlobal/interface/L1TGtObjectMapFwd.h"
#include "DataFormats/L1TGlobal/interface/L1TGtObjectMap.h"



namespace DataFormats_L1TGlobal {
  struct dictionary {

    GlobalAlgBlkBxCollection    uGtAlgBxColl;
    GlobalExtBlkBxCollection    uGtExtBxColl;    

    edm::Wrapper<GlobalAlgBlkBxCollection>    w_uGtAlgBxColl;
    edm::Wrapper<GlobalExtBlkBxCollection>    w_uGtExtBxColl;
    
    L1TGtObjectMap  uGtObjectMap;
    edm::Wrapper<L1TGtObjectMap> w_uGtObjectMap;

    L1TGtObjectMapRecord  uGtObjectMapRecord;
    edm::Wrapper<L1TGtObjectMapRecord> w_uGtObjectMapRecord;    
    
    
  };
}
