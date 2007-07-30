#ifndef DataFormats_Common_BasicHandle_h
#define DataFormats_Common_BasicHandle_h

/*----------------------------------------------------------------------
  
Handle: Non-owning "smart pointer" for reference to EDProducts and
their Provenances.

This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to EDProduct or Provenance is destroyed, use of the
Handle becomes undefined. There is no way to query the Handle to
discover if this has happened.

Handles can have:
  -- Product and Provenance pointers both null;
  -- Both pointers valid

To check validity, one can use the isValid() function.

$Id: BasicHandle.h,v 1.4 2007/05/29 21:23:37 wmtan Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <typeinfo>
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProductID.h"

namespace edm {
  class EDProduct;
  class BasicHandle {
  public:
    BasicHandle() :
      wrap_(0),
      prov_(0) {}

    BasicHandle(BasicHandle const& h) :
      wrap_(h.wrap_),
      prov_(h.prov_) {}

    BasicHandle(EDProduct const* prod, Provenance const* prov) :
      wrap_(prod), prov_(prov) {
    }

    ~BasicHandle() {}

    void swap(BasicHandle& other) {
      std::swap(wrap_, other.wrap_);
      std::swap(prov_, other.prov_);
    }

    
    BasicHandle& operator=(BasicHandle const& rhs) {
      BasicHandle temp(rhs);
      this->swap(temp);
      return *this;
    }

    bool isValid() const {
      return wrap_ && prov_;
    }

    EDProduct const* wrapper() const {
      return wrap_;
    }

    Provenance const* provenance() const {
      return prov_;
    }

    ProductID id() const {
      if (!prov_) {
        return ProductID();
      }
      return prov_->productID();
    }

  private:
    EDProduct const* wrap_;
    Provenance const* prov_;    
  };

  // Free swap function
  inline
  void
  swap(BasicHandle& a, BasicHandle& b) {
    a.swap(b);
  }
}

#endif
