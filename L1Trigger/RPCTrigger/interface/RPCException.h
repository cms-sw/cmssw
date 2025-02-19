#ifndef L1Trigger_RPCException_h
#define L1Trigger_RPCException_h
#if (__BCPLUSPLUS__ >= 0x0550)
#include <string>
#include "TException.h"
class RPCException: public TException {
   public:
     RPCException(std::string msg) : TException(msg) {};
 };
 
#elif defined _STAND_ALONE //_MSC_VER
#include <string>
#include "TException.h"
  class RPCException: public TException { //public __gc 
   public:
     RPCException(std::string msg) : TException(msg) {};
};

#else // not _STAND_ALONE
#include "FWCore/Utilities/interface/Exception.h"
  class RPCException: public cms::Exception {
    public:
      RPCException(std::string msg) : cms::Exception(msg) {};
};
#endif // _STAND_ALONE
 
#endif
