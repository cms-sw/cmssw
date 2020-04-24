#ifndef FWCore_Utilities_getAnyPtr_h
#define FWCore_Utilities_getAnyPtr_h

#include <cassert>
#include <memory>

namespace edm {
  template <typename T>
  inline
  std::unique_ptr<T> getAnyPtr(void* p, int offset) {
    // A union is used to avoid possible copies during the triple cast that would otherwise be needed. 	 
    // std::unique_ptr<T> edp(static_cast<T*>(static_cast<void *>(static_cast<unsigned char *>(p) + offset))); 	 
    union { 	 
      void* vp; 	 
      unsigned char* ucp; 	 
      T* tp; 	 
    } pointerUnion; 	 
    assert(p != nullptr);
    pointerUnion.vp = p; 	 
    pointerUnion.ucp += offset; 	 
    std::unique_ptr<T> tp(pointerUnion.tp);
    return(std::move(tp));
  }
}

#endif
