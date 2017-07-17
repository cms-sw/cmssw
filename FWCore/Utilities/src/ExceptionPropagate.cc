#include "FWCore/Utilities/interface/ExceptionPropagate.h"

namespace edm {
  namespace threadLocalException {
    static thread_local std::exception_ptr stdExceptionPtr;
    void setException(std::exception_ptr e) {
      stdExceptionPtr = e;
    }
    std::exception_ptr getException() {
      return stdExceptionPtr;
    }
  }

}

