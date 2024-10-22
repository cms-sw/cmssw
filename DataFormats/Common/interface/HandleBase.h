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

#include <cassert>
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Common/interface/HandleExceptionFactory.h"
#include <algorithm>

#include <memory>

namespace cms {
  class Exception;
}
namespace edm {
  class HandleBase {
  public:
    HandleBase() : product_(nullptr), prov_(nullptr) {}

    HandleBase(void const* prod, Provenance const* prov) : product_(prod), prov_(prov) {
      assert(prod);
      assert(prov);
    }

    ~HandleBase() {}

    void clear() {
      product_ = nullptr;
      prov_ = nullptr;
      whyFailedFactory_.reset();
    }

    void swap(HandleBase& other) {
      using std::swap;
      swap(product_, other.product_);
      std::swap(prov_, other.prov_);
      swap(whyFailedFactory_, other.whyFailedFactory_);
    }

    HandleBase& operator=(HandleBase const& rhs) {
      HandleBase temp(rhs);
      this->swap(temp);
      return *this;
    }

    bool isValid() const { return product_ && prov_; }

    bool failedToGet() const { return bool(whyFailedFactory_); }

    Provenance const* provenance() const { return prov_; }

    ProductID id() const;

    HandleBase(HandleBase const&) = default;

    ///Used when the attempt to get the data failed
    HandleBase(std::shared_ptr<HandleExceptionFactory const>&& iWhyFailed)
        : product_(), prov_(nullptr), whyFailedFactory_(iWhyFailed) {}

    HandleBase& operator=(HandleBase&& rhs) {
      product_ = rhs.product_;
      prov_ = rhs.prov_;
      whyFailedFactory_ = std::move(rhs.whyFailedFactory_);
      return *this;
    }

    std::shared_ptr<cms::Exception> whyFailed() const {
      if (whyFailedFactory_.get()) {
        return whyFailedFactory_->make();
      }
      return std::shared_ptr<cms::Exception>();
    }

    std::shared_ptr<HandleExceptionFactory const> const& whyFailedFactory() const { return whyFailedFactory_; }

    explicit operator bool() const { return isValid(); }

    bool operator!() const { return not isValid(); }

  protected:
    void const* productStorage() const;

  private:
    void const* product_;
    Provenance const* prov_;
    std::shared_ptr<HandleExceptionFactory const> whyFailedFactory_;
  };

  // Free swap function
  inline void swap(HandleBase& a, HandleBase& b) { a.swap(b); }
}  // namespace edm

#endif
