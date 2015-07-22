#ifndef DataFormats_Common_refcore_implementation_h
#define DataFormats_Common_refcore_implementation_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     refcore_implementation
// 
/**\class refcore_implementation refcore_implementation.h DataFormats/Common/interface/refcore_implementation.h

 Description: provide function implementations to use with both RefCore and RefCoreWithIndex 

 Usage:
    RefCore and RefCoreWithIndex are essentially the same except RefCoreWithIndex provides an additional index storage.
 Because of the exact way ROOT does storage, RefCoreWithIndex can't be made to inherit from RefCore (without causing meta
 data overhead). Therefore the implementations shared by the two classes are contained here.

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Nov  3 09:16:30 CDT 2011
//

// system include files
#ifndef __GCCXML__
#include <limits>
#include <cstdint>
#include <atomic>
#endif

// user include files

// forward declarations
namespace edm {
  namespace refcoreimpl {
    const unsigned short kTransientBit = 0x8000;
    const unsigned short kProcessIndexMask = 0x3FFF;
#ifndef __GCCXML__
    const std::uintptr_t kCacheIsProductPtrBit = 0x1;
    const std::uintptr_t kCacheIsProductPtrMask = std::numeric_limits<std::uintptr_t>::max() ^ kCacheIsProductPtrBit;

    inline bool cacheIsProductPtr( void const* iPtr) {
      return 0 == (reinterpret_cast<std::uintptr_t>(iPtr) & refcoreimpl::kCacheIsProductPtrBit);
    }

    inline void setCacheIsProductGetter( std::atomic<void const*> & ptr, EDProductGetter const* iGetter) {
      std::uintptr_t tmp = reinterpret_cast<std::uintptr_t>(iGetter); 
      tmp |=refcoreimpl::kCacheIsProductPtrBit; 
      ptr.store(reinterpret_cast<void const*>(tmp));
    }

    //Used by ROOT 5 I/O rule
    inline void setCacheIsProductGetter(void const*& ptr, EDProductGetter const* iGetter) {
      std::uintptr_t tmp = reinterpret_cast<std::uintptr_t>(iGetter); 
      tmp |=refcoreimpl::kCacheIsProductPtrBit;
      ptr = reinterpret_cast<void const*>(tmp);
    }


    inline void setCacheIsItem(std::atomic<void const*>& iCache, void const* iNewValue) {
      iCache = iNewValue;
    }
    
    inline bool tryToSetCacheItemForFirstTime(std::atomic<void const*>&iCache, void const* iNewValue) {
      auto cache = iCache.load();
      if(not cacheIsProductPtr(cache)) {
        return iCache.compare_exchange_strong(cache, iNewValue);
      }
      return false;
    }

    inline void const* productPtr(std::atomic<void const*> const& iCache) {
      auto tmp = iCache.load(); 
      return refcoreimpl::cacheIsProductPtr(tmp)?tmp:static_cast<void const*>(nullptr);
    }

    inline EDProductGetter const* productGetter(std::atomic<void const*> const& iCache) {
      auto tmp = iCache.load(); 
      return (!refcoreimpl::cacheIsProductPtr(tmp))? reinterpret_cast<EDProductGetter const*>(reinterpret_cast<std::uintptr_t>(tmp)&refcoreimpl::kCacheIsProductPtrMask):static_cast<EDProductGetter const*>(nullptr);
    }
#else
    bool cacheIsProductPtr( void const* iPtr) ;
    void setCacheIsItem(void const*&, void const*);
    void setCacheIsProductGetter(void const*&, EDProductGetter const*);
    void const* productPtr(void const*);
    EDProductGetter const* productGetter(void const*);
    bool tryToSetCacheItemForFirstTime(void const*& iCache, void const* iNewValue);
#endif

  }
}

#define ID_IMPL return ProductID(processIndex_ & refcoreimpl::kProcessIndexMask,productIndex_)

#define PRODUCTPTR_IMPL return refcoreimpl::productPtr(cachePtr_)

#define ISNONNULL_IMPL return isTransient() ? productPtr() != nullptr : id().isValid()

#define PRODUCTGETTER_IMPL return refcoreimpl::productGetter(cachePtr_)

#define ISTRANSIENT_IMPL return 0!=(processIndex_ & refcoreimpl::kTransientBit)

#define SETTRANSIENT_IMPL processIndex_ |=refcoreimpl::kTransientBit

#define SETCACHEISPRODUCTPTR_IMPL(_item_) refcoreimpl::setCacheIsItem(cachePtr_,_item_)

#define SETCACHEISPRODUCTGETTER_IMPL(_getter_) refcoreimpl::setCacheIsProductGetter(cachePtr_, _getter_)

#endif
