#ifndef CondCore_CondDB_CondDB_h
#define CondCore_CondDB_CondDB_h
#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/ORAWrapper.h"

namespace conddb {
  using Session = ora_wrapper::Session;
  using IOVEditor = ora_wrapper::IOVEditor;
  using IOVProxy = ora_wrapper::IOVProxy;
  using GTProxy = ora_wrapper::GTProxy;
}

#endif // CondCore_CondDB_CondDB_h
