//#include "CondCore/CondDB/interface/Session.h"
//#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/ORAWrapper.h"

namespace conddb {
  using Session = ora_wrapper::Session;
  using IOVEditor = ora_wrapper::IOVEditor;
  using IOVProxy = ora_wrapper::IOVProxy;
  using GTProxy = ora_wrapper::GTProxy;
  //template <typename T> using PayloadProxy = tmp::PayloadProxy<T>;
}

