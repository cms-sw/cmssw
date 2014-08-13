#ifndef DataFormats_Common_RefCore_h
#define DataFormats_Common_RefCore_h

/*----------------------------------------------------------------------
  
RefCore: The component of edm::Ref containing the product ID and product getter.

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/refcore_implementation.h"

#include <algorithm>
#include <typeinfo>

namespace edm {
  class RefCoreWithIndex;
  
  class RefCore {
    //RefCoreWithIndex is a specialization of RefCore done for performance
    // Since we need to freely convert one to the other the friendship is used
    friend class RefCoreWithIndex;
  public:
    RefCore() :  cachePtr_(0),processIndex_(0),productIndex_(0){}

    RefCore(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient);

    ProductID id() const {ID_IMPL;}

    /**If productPtr is not 0 then productGetter will be 0 since only one is available at a time */
    void const* productPtr() const {PRODUCTPTR_IMPL;}

    void setProductPtr(void const* prodPtr) const { 
      cachePtr_=prodPtr;
      setCacheIsProductPtr();
    }

    // Checks for null
    bool isNull() const {return !isNonnull(); }

    // Checks for non-null
    bool isNonnull() const {ISNONNULL_IMPL;}

    // Checks for null
    bool operator!() const {return isNull();}

    // Checks if collection is in memory or available
    // in the Event. No type checking is done.

    bool isAvailable() const;

    EDProductGetter const* productGetter() const {
      PRODUCTGETTER_IMPL;
    }

    void setProductGetter(EDProductGetter const* prodGetter) const;

    EDProduct const* getProductPtr(std::type_info const& type) const;

    void productNotFoundException(std::type_info const& type) const;

    void wrongTypeException(std::type_info const& expectedType, std::type_info const& actualType) const;

    void nullPointerForTransientException(std::type_info const& type) const;

    void swap(RefCore &);
    
    bool isTransient() const {ISTRANSIENT_IMPL;}

    int isTransientInt() const {return isTransient() ? 1 : 0;}

    void pushBackItem(RefCore const& productToBeInserted, bool checkPointer);

 private:
    RefCore(void const* iCache, ProcessIndex iProcessIndex, ProductIndex iProductIndex):
    cachePtr_(iCache), processIndex_(iProcessIndex), productIndex_(iProductIndex) {}
    void setId(ProductID const& iId);
    void setTransient() {SETTRANSIENT_IMPL;}
    void setCacheIsProductPtr() const {SETCACHEISPRODUCTPTR_IMPL;}
    void unsetCacheIsProductPtr() const {UNSETCACHEISPRODUCTPTR_IMPL;}
    bool cacheIsProductPtr() const {CACHEISPRODUCTPTR_IMPL;}

    
    mutable void const* cachePtr_;               // transient
    //The following are what is stored in a ProductID
    // the high two bits of processIndex are used to store info on
    // if this is transient and if the cachePtr_ is storing the productPtr
    //If the type or order of the member data is changes you MUST also update
    // the custom streamer in RefCoreStreamer.cc and RefCoreWithIndex
    mutable ProcessIndex processIndex_;
    ProductIndex productIndex_;

  };

  inline
  bool
  operator==(RefCore const& lhs, RefCore const& rhs) {
    return lhs.isTransient() == rhs.isTransient() && (lhs.isTransient() ? lhs.productPtr() == rhs.productPtr() : lhs.id() == rhs.id());
  }

  inline
  bool
  operator!=(RefCore const& lhs, RefCore const& rhs) {
    return !(lhs == rhs);
  }

  inline
  bool
  operator<(RefCore const& lhs, RefCore const& rhs) {
    return lhs.isTransient() ? (rhs.isTransient() ? lhs.productPtr() < rhs.productPtr() : false) : (rhs.isTransient() ? true : lhs.id() < rhs.id());
  }

  inline 
  void
  RefCore::swap(RefCore & other) {
    std::swap(processIndex_, other.processIndex_);
    std::swap(productIndex_, other.productIndex_);
    std::swap(cachePtr_, other.cachePtr_);
  }

  inline void swap(edm::RefCore & lhs, edm::RefCore & rhs) {
    lhs.swap(rhs);
  }
}

#endif
