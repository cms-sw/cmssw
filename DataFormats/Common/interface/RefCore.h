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
    RefCore() :  cachePtr_(0),processIndex_(0),productIndex_(0), transient_() {}

    RefCore(ProductID const& theId, void const* prodPtr, EDProductGetter const* prodGetter, bool transient);

    ProductID id() const {return ProductID(processIndex_,productIndex_);}

    /**If productPtr is not 0 then productGetter will be 0 since only one is available at a time */
    void const* productPtr() const {return cacheIsProductPtr()?cachePtr_:static_cast<void const*>(0);}

    void setProductPtr(void const* prodPtr) const { 
      cachePtr_=prodPtr;
      setCacheIsProductPtr(true);
    }

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
      return (!cacheIsProductPtr())? static_cast<EDProductGetter const*>(cachePtr_):
      static_cast<EDProductGetter const*>(0);
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

    struct CheckTransientOnWrite {
      explicit CheckTransientOnWrite(bool iValue=false, bool iIsProductPtr=false): 
      transient_(iValue),
      cacheIsProductPtr_(iIsProductPtr){}
      bool transient_;
      mutable bool cacheIsProductPtr_; //transient
    };
 private:
    void setId(ProductID const& iId) {
      processIndex_ = iId.processIndex();
      productIndex_ = iId.productIndex();
    }
    void setTransient() {transient_.transient_ = true;}
    void setCacheIsProductPtr(bool iState) const {transient_.cacheIsProductPtr_=iState;}
    bool cacheIsProductPtr() const {
      return transient_.cacheIsProductPtr_;
    }

    
    mutable void const* cachePtr_;               // transient
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
    std::swap(cachePtr_, other.cachePtr_);
    std::swap(transient_.transient_, other.transient_.transient_);
    std::swap(transient_.cacheIsProductPtr_, other.transient_.cacheIsProductPtr_);
  }

  inline void swap(edm::RefCore & lhs, edm::RefCore & rhs) {
    lhs.swap(rhs);
  }
}

#endif
