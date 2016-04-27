#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include <vector>

namespace DataFormats_L1TGlobal {
  struct dictionary {

    GlobalAlgBlkBxCollection    uGtAlgBxColl;
    GlobalExtBlkBxCollection    uGtExtBxColl;

    edm::Wrapper<GlobalAlgBlkBxCollection>    w_uGtAlgBxColl;
    edm::Wrapper<GlobalExtBlkBxCollection>    w_uGtExtBxColl;

    std::vector<GlobalAlgBlk> v_uGtAlgBx;
    std::vector<GlobalExtBlk> v_uGtExtBx;
  };
}
