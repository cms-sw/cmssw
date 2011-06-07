#ifndef DataFormats_Common_RefCore_h
#define DataFormats_Common_RefCore_h

/*----------------------------------------------------------------------
  
RefCore: The component of edm::Ref containing the product ID and product getter.

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include <algorithm>
#include <typeinfo>

namespace edm {
  class RefCore {
  public:
    RefCore() :  prodPtr_(0),prodGetter_(0),clientCache_(0), processIndex_(0),productIndex_(0), transient_() {}

    RefCore(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient);

    ProductID id() const {return ProductID(processIndex_,productIndex_);}

    void const* productPtr() const {return prodPtr_;}

    void setProductPtr(void const* prodPtr) const { prodPtr_=prodPtr;}

    // Checks for null
    bool isNull() const {return !isNonnull(); }

    // Checks for non-null
    bool isNonnull() const {return isTransient() ? productPtr() != 0 : id().isValid();}

    // Checks for null
    bool operator!() const {return isNull();}

    // Checks if collection is in memory or available
    // in the Event. No type checking is done.

    bool isAvailable() const;

    EDProductGetter const* productGetter() const {
      return prodGetter_;
    }

    void setProductGetter(EDProductGetter const* prodGetter) const;

    WrapperHolder getProductPtr(std::type_info const& type) const;

    void productNotFoundException(std::type_info const& type) const;

    void wrongTypeException(std::type_info const& expectedType, std::type_info const& actualType) const;

    void nullPointerForTransientException(std::type_info const& type) const;

    void swap(RefCore &);
    
    bool isTransient() const {return transient_.transient_;}

    int isTransientInt() const {return transient_.transient_ ? 1 : 0;}

    void pushBackItem(RefCore const& productToBeInserted, bool checkPointer);

    //the client ptr allows templated classes which hold a RefCore to use for a transient cache
    void const* clientCache() const { return clientCache_;}
    void const*& mutableClientCache() { return clientCache_;}
    
    struct CheckTransientOnWrite {
      explicit CheckTransientOnWrite(bool iValue=false): transient_(iValue) {}
      bool transient_;
    };
 private:
    void setId(ProductID const& iId) {
      processIndex_ = iId.processIndex();
      productIndex_ = iId.productIndex();
    }
    void setTransient() {transient_.transient_ = true;}

    mutable void const* prodPtr_;               // transient
    mutable EDProductGetter const* prodGetter_; // transient
    mutable void const* clientCache_;           // transient
    //The following are what is stored in a ProductID
    ProcessIndex processIndex_;
    ProductIndex productIndex_;
    CheckTransientOnWrite transient_;           // transient
    
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
    std::swap(prodPtr_, other.prodPtr_);
    std::swap(prodGetter_, other.prodGetter_);
    std::swap(clientCache_,other.clientCache_);
    std::swap(transient_.transient_, other.transient_.transient_);
  }

  inline void swap(edm::RefCore & lhs, edm::RefCore & rhs) {
    lhs.swap(rhs);
  }
}

#endif
