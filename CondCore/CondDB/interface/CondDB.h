//#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/tmp.h"
#include "CondCore/CondDB/interface/PayloadProxy.h"

namespace conddb {
  using Session = tmp::Session;
  using IOVEditor = tmp::IOVEditor;
  using IOVProxy = tmp::IOVProxy;
  using GTProxy = tmp::GTProxy;
  //template <typename T> using PayloadProxy = tmp::PayloadProxy<T>;
}

