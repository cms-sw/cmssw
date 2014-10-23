#ifndef IOPool_Common_getWrapperBasePtr_h
#define IOPool_Common_getWrapperBasePtr_h

#include "DataFormats/Common/interface/WrapperBase.h"
#include "FWCore/Utilities//interface/getAnyPtr.h"

namespace edm {
  inline
  std::unique_ptr<WrapperBase> getWrapperBasePtr(void* p, int offset) {
    return(std::move(getAnyPtr<WrapperBase>(p, offset)));
  }
}

#endif
