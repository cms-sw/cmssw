#ifndef Common_RefCore_h
#define Common_RefCore_h

/*----------------------------------------------------------------------
  
RefCore: The component of edm::Ref containing the product ID and product getter.

$Id: RefCore.h,v 1.6 2006/10/28 02:07:36 wmtan Exp $

----------------------------------------------------------------------*/
#include <typeinfo>
#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

namespace edm {
  class EDProductGetter;
  class RefCore {
  public:
    RefCore() : id_(), prodPtr_(0), prodGetter_(0) {}
    RefCore(ProductID const& theId, void const *prodPtr, EDProductGetter const* prodGetter) :
        id_(theId), prodPtr_(prodPtr), prodGetter_(prodGetter) {
      if (theId == ProductID()) {badID();}
    }
    ~RefCore() {}

    ProductID id() const {return id_;}

    void const* productPtr() const {return prodPtr_;}

    void setProductPtr(void const* prodPtr) const {prodPtr_ = prodPtr;}

    // Checks for null
    bool isNull() const {return id() == ProductID();}

    // Checks for non-null
    bool isNonnull() const {return !isNull();}

    // Checks for null
    bool operator!() const {return isNull();}

    EDProductGetter const* productGetter() const {return prodGetter_;}

    void setProductGetter(EDProductGetter const* prodGetter) const {prodGetter_ = prodGetter;}

    void nullID() const;

  private:
    void badID() const;

  private:
    ProductID id_;
    mutable void const *prodPtr_; // transient
    mutable EDProductGetter const* prodGetter_; // transient
  };

  inline
  bool
  operator==(RefCore const& lhs, RefCore const& rhs) {
    return lhs.id() == rhs.id();
  }

  inline
  bool
  operator!=(RefCore const& lhs, RefCore const& rhs) {
    return !(lhs == rhs);
  }

  inline
  bool
  operator<(RefCore const& lhs, RefCore const& rhs) {
    return lhs.id() < rhs.id();
  }

  void wrongRefType(std::string const& found, std::string const& requested);

  template <typename T>
  T const* getProduct_(RefCore const& product) {
    if (product.isNull()) {
      product.nullID();
    }
    if (!product.productGetter()) {
      product.setProductGetter(EDProductGetter::instance());
      assert(product.productGetter());
    }
    Wrapper<T> const* edpw = dynamic_cast<Wrapper<T> const*>(product.productGetter()->getIt(product.id()));
    if (edpw == 0) {
      wrongRefType(typeid(product.productGetter()->getIt(product.id())).name(), typeid(T).name());
    }
    return edpw->product();
  }

  template <typename T>
  inline
  T const* getProduct(RefCore const& product) {
    T const* p = static_cast<T const *>(product.productPtr());
    return(p != 0 ? p : getProduct_<T>(product));
  }

  void checkProduct(RefCore const& prod, RefCore & product);

}

#endif
