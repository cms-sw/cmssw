#ifndef DataFormats_Common_RefCore_h
#define DataFormats_Common_RefCore_h

/*----------------------------------------------------------------------
  
RefCore: The component of edm::Ref containing the product ID and product getter.

$Id: RefCore.h,v 1.20 2008/02/15 05:57:03 wmtan Exp $

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include <algorithm>

namespace edm {
  class EDProduct;
  class RefCore {
  public:
    RefCore() : id_(), prodPtr_(0), prodGetter_(0), transient_(false) {}

    RefCore(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient) :
      id_(theId), 
      prodPtr_(prodPtr), 
      prodGetter_(prodGetter),
      transient_(transient) { }

    ProductID id() const {return id_;}

    void const* productPtr() const {return prodPtr_;}

    void setProductPtr(void const* prodPtr) const {prodPtr_ = prodPtr;}

    // Checks for null
    bool isNull() const {return !isNonnull(); }

    // Checks for non-null
    bool isNonnull() const {return isTransient() ? productPtr() != 0 : id_.isValid();}

    // Checks for null
    bool operator!() const {return isNull();}

    // Checks if collection is in memory or available
    // in the Event. No type checking is done.

    bool isAvailable() const;

    EDProductGetter const* productGetter() const {
      if (!prodGetter_) setProductGetter(EDProductGetter::instance());
      return prodGetter_;
    }

    void setProductGetter(EDProductGetter const* prodGetter) const;

    EDProduct const* getProductPtr() const;

    void swap(RefCore &);
    
    bool isTransient() const {return transient_;}

    int isTransientInt() const {return transient_ ? 1 : 0;}

    void pushBackItem(RefCore const& productToBeInserted, bool checkPointer);

 private:
    void setId(ProductID const& iId) {id_ = iId;}
    void setTransient() {transient_ = true;}

    ProductID id_;
    mutable void const* prodPtr_;               // transient
    mutable EDProductGetter const* prodGetter_; // transient
    bool transient_;				// transient
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
    return lhs.isTransient() ? (rhs.isTransient() ? lhs.productPtr() < rhs.productPtr() : false ) : (rhs.isTransient() ? true : lhs.id() < rhs.id());
  }

  inline 
  void
  RefCore::swap( RefCore & other ) {
    std::swap(id_, other.id_);
    std::swap(prodPtr_, other.prodPtr_);
    std::swap(prodGetter_, other.prodGetter_);
    std::swap(transient_, other.transient_);
  }

  inline void swap(edm::RefCore & lhs, edm::RefCore & rhs) {
    lhs.swap(rhs);
  }
}

#endif
