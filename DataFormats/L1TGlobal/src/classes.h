#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TGlobal/interface/AlgBlk.h"
#include "DataFormats/L1TGlobal/interface/ExtBlk.h"
#include "DataFormats/L1TGlobal/interface/RecBlk.h"

namespace DataFormats_L1Trigger {
  struct dictionary {

    AlgBxCollection    uGtAlgBxColl;
    ExtBxCollection    uGtExtBxColl;
    RecBxCollection    uGtRecBxColl;    

    edm::Wrapper<AlgBxCollection>    w_uGtAlgBxColl;
    edm::Wrapper<ExtBxCollection>    w_uGtExtBxColl;
    edm::Wrapper<RecBxCollection>    w_uGtRecBxColl;   
  };
}
