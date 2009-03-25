#ifndef DataFormats_Common_HandleBase_h
#define DataFormats_Common_HandleBase_h

/*----------------------------------------------------------------------
  
Handle: Non-owning "smart pointer" for reference to products and
their provenances.

This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to product or provenance is destroyed, use of the
Handle becomes undefined. There is no way to query the Handle to
discover if this has happened.

Handles can have:
  -- Product and Provenance pointers both null;
  -- Both pointers valid

To check validity, one can use the isValid() function.

If failedToGet() returns true then the requested data is not available
If failedToGet() returns false but isValid() is also false then no attempt 
  to get data has occurred

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include <boost/shared_ptr.hpp>

namespace cms {
  class Exception;
}
namespace edm {
  class HandleBase {
  public:
    HandleBase() :
      product_(0),
      prov_(0) {}

    HandleBase(void const* prod, Provenance const* prov) :
      product_(prod), prov_(prov) {
      assert(prod);
      assert(prov);
    }

    ///Used when the attempt to get the data failed
    HandleBase(boost::shared_ptr<cms::Exception> const& iWhyFailed) :
    product_(),
    prov_(0),
    whyFailed_(iWhyFailed) {}
    
    ~HandleBase() {}

    void clear() {
      product_ = 0;
      prov_ = 0;
      whyFailed_.reset();
    }

    void swap(HandleBase& other) {
      using std::swap;
      swap(product_, other.product_);
      std::swap(prov_, other.prov_);
      swap(whyFailed_, other.whyFailed_);
    }
    
    HandleBase& operator=(HandleBase const& rhs) {
      HandleBase temp(rhs);
      this->swap(temp);
      return *this;
    }

    bool isValid() const {
      return product_ && prov_;
    }

    bool failedToGet() const {
      return 0 != whyFailed_.get();
    }
    
    Provenance const* provenance() const {
      return prov_;
    }

    ProductID id() const;

    boost::shared_ptr<cms::Exception> whyFailed() const {
      return whyFailed_;
    }

  protected:

    void const* productStorage() const;

  private:
    void const* product_;
    Provenance const* prov_;
    boost::shared_ptr<cms::Exception> whyFailed_;
  };

  // Free swap function
  inline
  void
  swap(HandleBase& a, HandleBase& b) {
    a.swap(b);
  }
}

#endif
