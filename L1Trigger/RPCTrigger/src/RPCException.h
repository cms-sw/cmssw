#ifndef RPCExceptionH
#define RPCExceptionH
#if (__BCPLUSPLUS__ >= 0x0550)
#include <string>
#include "TException.h"
class L1RpcException: public TException {
   public:
     L1RpcException(std::string msg) : TException(msg) {};
 };
 
#elif defined _STAND_ALONE //_MSC_VER
#include <string>
#include "TException.h"
  class L1RpcException: public TException { //public __gc 
   public:
     L1RpcException(std::string msg) : TException(msg) {};
};

#else // not _STAND_ALONE
#include "FWCore/Utilities/interface/Exception.h"
  class L1RpcException: public cms::Exception {
    public:
      L1RpcException(std::string msg) : cms::Exception(msg) {};
};
#endif // _STAND_ALONE
 
#endif
