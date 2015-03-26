#ifndef FWCore_Utilities_ExceptionPropagate_h
#define FWCore_Utilities_ExceptionPropagate_h

#include <exception>

namespace edm {
  namespace threadLocalException {
    void setException(std::exception_ptr e);
    std::exception_ptr getException();
  }
}

#endif
