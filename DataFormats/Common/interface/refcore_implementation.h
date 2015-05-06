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
#include <limits>
#include <cstdint>

// user include files

// forward declarations
namespace edm {
  namespace refcoreimpl {
    const unsigned short kTransientBit = 0x8000;
    const std::uintptr_t kCacheIsProductPtrBit = 0x1;
    const std::uintptr_t kCacheIsProductPtrMask = std::numeric_limits<std::uintptr_t>::max() ^ kCacheIsProductPtrBit;
    const unsigned short kProcessIndexMask = 0x3FFF;
    
    inline void setCacheIsProductGetter( void const * & ptr) {
      std::uintptr_t tmp = reinterpret_cast<std::uintptr_t>(ptr); tmp |=refcoreimpl::kCacheIsProductPtrBit; ptr = reinterpret_cast<void const*>(tmp);
    }

    inline void unsetCacheIsProductGetter( void const * & ptr) {
      std::uintptr_t tmp = reinterpret_cast<std::uintptr_t>(ptr); tmp &=refcoreimpl::kCacheIsProductPtrMask; ptr = reinterpret_cast<void const*>(tmp);
    }

  }
  
}

#define ID_IMPL return ProductID(processIndex_ & refcoreimpl::kProcessIndexMask,productIndex_)

#define PRODUCTPTR_IMPL return cacheIsProductPtr()?cachePtr_:static_cast<void const*>(nullptr)

#define ISNONNULL_IMPL return isTransient() ? productPtr() != nullptr : id().isValid()

#define PRODUCTGETTER_IMPL return (!cacheIsProductPtr())? reinterpret_cast<EDProductGetter const*>(reinterpret_cast<std::uintptr_t>(cachePtr_)&refcoreimpl::kCacheIsProductPtrMask):static_cast<EDProductGetter const*>(nullptr)

#define ISTRANSIENT_IMPL return 0!=(processIndex_ & refcoreimpl::kTransientBit)

#define SETTRANSIENT_IMPL processIndex_ |=refcoreimpl::kTransientBit

#define SETCACHEISPRODUCTPTR_IMPL refcoreimpl::unsetCacheIsProductGetter(cachePtr_)

#define SETCACHEISPRODUCTGETTER_IMPL refcoreimpl::setCacheIsProductGetter(cachePtr_)

#define CACHEISPRODUCTPTR_IMPL return 0 == (reinterpret_cast<std::uintptr_t>(cachePtr_) & refcoreimpl::kCacheIsProductPtrBit)

#endif
