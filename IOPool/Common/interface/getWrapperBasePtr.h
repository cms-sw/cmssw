#ifndef IOPool_Common_getWrapperBasePtr_h
#define IOPool_Common_getWrapperBasePtr_h

#include <cassert>
#include <memory>
#include "TClass.h"
#include "DataFormats/Common/interface/WrapperBase.h"

namespace edm {
  inline
  std::unique_ptr<WrapperBase> getWrapperBasePtr(void* p, int offset) {
    // A union is used to avoid possible copies during the triple cast that would otherwise be needed. 	 
    // std::unique_ptr<WrapperBase> edp(static_cast<WrapperBase *>(static_cast<void *>(static_cast<unsigned char *>(p) + branchInfo.offsetToWrapperBase_))); 	 
    union { 	 
      void* vp; 	 
      unsigned char* ucp; 	 
      WrapperBase* edp; 	 
    } pointerUnion; 	 
    assert(p != nullptr);
    pointerUnion.vp = p; 	 
    pointerUnion.ucp += offset; 	 
    std::unique_ptr<WrapperBase> edp(pointerUnion.edp);
    return(std::move(edp));
  }
}

#endif
