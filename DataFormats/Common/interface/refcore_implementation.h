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
// $Id$
//

// system include files

// user include files

// forward declarations
namespace edm {
  namespace refcoreimpl {
    const unsigned short kTransientBit = 0x8000;
    const unsigned short kCacheIsProductPtrBit = 0x4000;
    const unsigned short kCacheIsProductPtrMask = 0xBFFF;
    const unsigned short kProcessIndexMask = 0x3FFF;
  }
}

#define ID_IMPL return ProductID(processIndex_ & refcoreimpl::kProcessIndexMask,productIndex_)

#define PRODUCTPTR_IMPL return cacheIsProductPtr()?cachePtr_:static_cast<void const*>(0)

#define ISNONNULL_IMPL return isTransient() ? productPtr() != 0 : id().isValid()

#define PRODUCTGETTER_IMPL return (!cacheIsProductPtr())? static_cast<EDProductGetter const*>(cachePtr_):static_cast<EDProductGetter const*>(0)

#define ISTRANSIENT_IMPL return 0!=(processIndex_ & refcoreimpl::kTransientBit)

#define SETTRANSIENT_IMPL processIndex_ |=refcoreimpl::kTransientBit

#define SETCACHEISPRODUCTPTR_IMPL processIndex_ |=refcoreimpl::kCacheIsProductPtrBit

#define UNSETCACHEISPRODUCTPTR_IMPL processIndex_ &=refcoreimpl::kCacheIsProductPtrMask

#define CACHEISPRODUCTPTR_IMPL return 0 != (processIndex_ & refcoreimpl::kCacheIsProductPtrBit)

#endif
