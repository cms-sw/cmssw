#ifndef CondCore_CondDB_CondDB_h
#define CondCore_CondDB_CondDB_h
#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/ORAWrapper.h"

namespace cond {

  namespace db {
    using Session =  cond::ora_wrapper::Session;
    using Transaction =  cond::ora_wrapper::Transaction;
    using IOVEditor =  cond::ora_wrapper::IOVEditor;
    using IOVProxy =  cond::ora_wrapper::IOVProxy;
    using GTProxy =  cond::ora_wrapper::GTProxy;
    using Exception = cond::persistency::Exception; 
  }
}

#endif // CondCore_CondDB_CondDB_h
