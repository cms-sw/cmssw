#ifndef CondCore_CondDB_CondDB_h
#define CondCore_CondDB_CondDB_h
#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/ORAWrapper.h"

namespace cond {

  namespace db {
    using Session =  cond::ora_wrapper::Session;
    using IOVEditor =  cond::ora_wrapper::IOVEditor;
    using IOVProxy =  cond::ora_wrapper::IOVProxy;
    using GTProxy =  cond::ora_wrapper::GTProxy;
  }
}

#endif // CondCore_CondDB_CondDB_h
