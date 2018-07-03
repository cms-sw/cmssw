#ifndef DataFormats_Common_BasicHandle_h
#define DataFormats_Common_BasicHandle_h

/*----------------------------------------------------------------------

Handle: Shared "smart pointer" for reference to EDProducts and
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

If failedToGet() returns true then the requested data is not available
If failedToGet() returns false but isValid() is also false then no attempt
  to get data has occurred

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/HandleExceptionFactory.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#include <memory>

namespace cms {
  class Exception;
}

namespace edm {
  class WrapperBase;
  template <typename T> class Wrapper;

  class BasicHandle {
  public:
    BasicHandle() :
      product_(),
      prov_(nullptr) {}

    BasicHandle(BasicHandle const& h) :
      product_(h.product_),
      prov_(h.prov_),
      whyFailedFactory_(h.whyFailedFactory_){}

    BasicHandle(BasicHandle &&h) = default;
    
    BasicHandle(WrapperBase const* iProd, Provenance const* iProv) :
      product_(iProd),
      prov_(iProv) {
    }

    ///Used when the attempt to get the data failed
    BasicHandle(std::shared_ptr<HandleExceptionFactory> const& iWhyFailed):
    product_(),
    prov_(nullptr),
    whyFailedFactory_(iWhyFailed) {}

    ~BasicHandle() {}

    void swap(BasicHandle& other) {
      using std::swap;
      swap(product_, other.product_);
      std::swap(prov_, other.prov_);
      swap(whyFailedFactory_,other.whyFailedFactory_);
    }

    BasicHandle& operator=(BasicHandle const& rhs) {
      BasicHandle temp(rhs);
      this->swap(temp);
      return *this;
    }

    bool isValid() const {
      return product_ && prov_;
    }

    bool failedToGet() const {
      return bool(whyFailedFactory_);
    }

    WrapperBase const* wrapper() const {
      return product_;
    }

    Provenance const* provenance() const {
      return prov_;
    }

    ProductID id() const {
      return prov_->productID();
    }

    std::shared_ptr<cms::Exception> whyFailed() const {
      return whyFailedFactory_->make();
    }
    
    std::shared_ptr<HandleExceptionFactory> const& whyFailedFactory() const {
      return whyFailedFactory_;
    }
    
    std::shared_ptr<HandleExceptionFactory>& whyFailedFactory()  {
      return whyFailedFactory_;
    }

    void clear() {
      product_ = nullptr;
      prov_ = nullptr;
      whyFailedFactory_.reset();
    }

  private:
    WrapperBase const* product_;
    Provenance const* prov_;
    std::shared_ptr<HandleExceptionFactory> whyFailedFactory_;
  };

  // Free swap function
  inline
  void
  swap(BasicHandle& a, BasicHandle& b) {
    a.swap(b);
  }
}

#endif
