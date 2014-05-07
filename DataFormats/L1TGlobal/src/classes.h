#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TGlobal/interface/AlgBlk.h"
#include "DataFormats/L1TGlobal/interface/ExtBlk.h"

namespace DataFormats_L1Trigger {
  struct dictionary {

    AlgBxCollection    uGtAlgBxColl;
    ExtBxCollection    uGtExtBxColl;

    edm::Wrapper<AlgBxCollection>    w_uGtAlgBxColl;
    edm::Wrapper<ExtBxCollection>    w_uGtExtBxColl;
  };
}
