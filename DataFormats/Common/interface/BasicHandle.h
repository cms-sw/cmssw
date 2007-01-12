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

$Id: BasicHandle.h,v 1.12 2007/01/11 23:39:19 paterno Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <stdexcept>
#include <typeinfo>

#include "DataFormats/Common/interface/Provenance.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  class EDProduct;
  class BasicHandle {
  public:
    // Default constructed handles are invalid.
    BasicHandle() :
      wrap_(0),
      prov_(0) {}

    BasicHandle(BasicHandle const& h) :
      wrap_(h.wrap_),
      prov_(h.prov_) {}

    BasicHandle(EDProduct const* prod, Provenance const* prov) :
      wrap_(prod), prov_(prov) {
      assert(wrap_);
      assert(prov_);
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
    // Should we throw if the pointer is null?
      return wrap_;
    }

    Provenance const* provenance() const {
    // Should we throw if the pointer is null?
      return prov_;
    }

    ProductID id() const {
      if (!prov_) 
	throw Exception(errors::NullPointerError)
	  << "Attempt to get ID from an invalid BasicHandle\n";
      return prov_->event.productID_;
    }

  private:
    EDProduct const* wrap_;
    Provenance const* prov_;    
  };

  // Free swap function
  inline
  void
  swap(BasicHandle& a, BasicHandle& b) 
  {
    a.swap(b);
  }
}

#endif
