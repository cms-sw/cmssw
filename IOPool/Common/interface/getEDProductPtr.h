#ifndef IOPool_Common_getEDProductPtr_h
#define IOPool_Common_getEDProductPtr_h

#include <cassert>
#include <memory>
#include "TClass.h"
#include "DataFormats/Common/interface/EDProduct.h"

namespace edm {
  inline
  std::unique_ptr<EDProduct> getEDProductPtr(void* p, int offset) {
    // A union is used to avoid possible copies during the triple cast that would otherwise be needed. 	 
    // std::unique_ptr<EDProduct> edp(static_cast<EDProduct *>(static_cast<void *>(static_cast<unsigned char *>(p) + branchInfo.offsetToEDProduct_))); 	 
    union { 	 
      void* vp; 	 
      unsigned char* ucp; 	 
      EDProduct* edp; 	 
    } pointerUnion; 	 
    assert(p != nullptr);
    pointerUnion.vp = p; 	 
    pointerUnion.ucp += offset; 	 
    std::unique_ptr<EDProduct> edp(pointerUnion.edp);
    return(std::move(edp));
  }
}

#endif
