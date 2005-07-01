#ifndef EDM_BASICHANDLE_H
#define EDM_BASICHANDLE_H

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

$Id: Handle.h,v 1.3 2005/06/28 04:46:02 jbk Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <stdexcept>
#include <typeinfo>

#include "boost/utility/enable_if.hpp"
#include "boost/type_traits.hpp"

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/FWUtilities/interface/EDMException.h"

namespace edm {
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

    EDP_ID id() const {return wrap_->id();}

  private:
    EDProduct const* wrap_;
    Provenance const* prov_;    
  };
}

#endif
