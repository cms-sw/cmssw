#ifndef DataFormats_Common_RefCoreWithIndex_h
#define DataFormats_Common_RefCoreWithIndex_h

/*----------------------------------------------------------------------
  
RefCoreWithIndex: The component of edm::Ref containing the product ID and product getter and the index into the collection.
    This class is a specialization of RefCore and forwards most of the
    implementation to RefCore.

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/traits.h"
#include "DataFormats/Common/interface/refcore_implementation.h"
#include "DataFormats/Common/interface/RefCore.h"

#include <algorithm>
#include <typeinfo>

namespace edm {  
  class RefCoreWithIndex {
  public:
    RefCoreWithIndex() :  cachePtr_(0),processIndex_(0),productIndex_(0),elementIndex_(edm::key_traits<unsigned int>::value) {}

    RefCoreWithIndex(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient, unsigned int elementIndex);

    RefCoreWithIndex(RefCore const& iCore, unsigned int);
    
    ProductID id() const {ID_IMPL;}

    /**If productPtr is not 0 then productGetter will be 0 since only one is available at a time */
    void const* productPtr() const {PRODUCTPTR_IMPL;}

    void setProductPtr(void const* prodPtr) const { 
      cachePtr_=prodPtr;
      setCacheIsProductPtr();
    }

    
    unsigned int index() const { return elementIndex_;}
    
    // Checks for null
    bool isNull() const {return !isNonnull(); }

    // Checks for non-null
    bool isNonnull() const {ISNONNULL_IMPL;}

    // Checks for null
    bool operator!() const {return isNull();}

    // Checks if collection is in memory or available
    // in the Event. No type checking is done.

    bool isAvailable() const {
      return toRefCore().isAvailable();
    }

    //Convert to an equivalent RefCore. Needed for Ref specialization.
    RefCore const& toRefCore() const {
      return *reinterpret_cast<const RefCore*>(this);
    }
    
    EDProductGetter const* productGetter() const {
      PRODUCTGETTER_IMPL;
    }

    void setProductGetter(EDProductGetter const* prodGetter) const {
      toRefCore().setProductGetter(prodGetter);
    }

    WrapperHolder getProductPtr(std::type_info const& type) const {
      return toRefCore().getProductPtr(type);
    }

    void productNotFoundException(std::type_info const& type) const {
      toRefCore().productNotFoundException(type);
    }

    void wrongTypeException(std::type_info const& expectedType, std::type_info const& actualType) const {
      toRefCore().wrongTypeException(expectedType,actualType);
    }

    void nullPointerForTransientException(std::type_info const& type) const {
      toRefCore().nullPointerForTransientException(type);
    }

    void swap(RefCoreWithIndex &);
    
    bool isTransient() const {ISTRANSIENT_IMPL;}

    int isTransientInt() const {return isTransient() ? 1 : 0;}

    void pushBackItem(RefCoreWithIndex const& productToBeInserted, bool checkPointer) {
      toUnConstRefCore().pushBackItem(productToBeInserted.toRefCore(),checkPointer);
    }

 private:
    RefCore& toUnConstRefCore(){
      return *reinterpret_cast<RefCore*>(this);
    }

    void setId(ProductID const& iId) {
      toUnConstRefCore().setId(iId);
    }
    void setTransient() {SETTRANSIENT_IMPL;}
    void setCacheIsProductPtr() const {SETCACHEISPRODUCTPTR_IMPL;}
    void unsetCacheIsProductPtr() const {UNSETCACHEISPRODUCTPTR_IMPL;}
    bool cacheIsProductPtr() const {CACHEISPRODUCTPTR_IMPL;}

    //NOTE: the order MUST remain the same as a RefCore
    // since we play tricks to allow a pointer to a RefCoreWithIndex
    // to be the same as a pointer to a RefCore
    mutable void const* cachePtr_;               // transient
    //The following are what is stored in a ProductID
    // the high two bits of processIndex are used to store info on
    // if this is transient and if the cachePtr_ is storing the productPtr
    mutable ProcessIndex processIndex_;
    ProductIndex productIndex_;
    unsigned int elementIndex_;

  };

  inline 
  void
  RefCoreWithIndex::swap(RefCoreWithIndex & other) {
    std::swap(processIndex_, other.processIndex_);
    std::swap(productIndex_, other.productIndex_);
    std::swap(cachePtr_, other.cachePtr_);
    std::swap(elementIndex_,other.elementIndex_);
  }

  inline void swap(edm::RefCoreWithIndex & lhs, edm::RefCoreWithIndex & rhs) {
    lhs.swap(rhs);
  }
}

#endif
