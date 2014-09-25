#ifndef DataFormats_Common_OutputHandle_h
#define DataFormats_Common_OutputHandle_h

/*----------------------------------------------------------------------
  
Handle: Non-owning "smart pointer" for reference to EDProducts and
their Provenances.

This is a very preliminary version, and lacks safety features and
elegance.

If the pointed-to object or provenance destroyed, use of the
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

#include "DataFormats/Provenance/interface/ProductProvenance.h"

#include <memory>

namespace cms {
  class Exception;
}

namespace edm {
  class BranchDescription;
  class WrapperBase;
  class OutputHandle {
  public:
    OutputHandle() :
      product_(nullptr),
      desc_(nullptr),
      productProvenance_(0) {}

    OutputHandle(OutputHandle const& h) :
      product_(h.product_),
      desc_(h.desc_),
      productProvenance_(h.productProvenance_),
      whyFailed_(h.whyFailed_){}

    OutputHandle(WrapperBase const* product, BranchDescription const* desc, ProductProvenance* productProvenance) :
      product_(product),
      desc_(desc),
      productProvenance_(productProvenance) {}

    ///Used when the attempt to get the data failed
    OutputHandle(std::shared_ptr<cms::Exception> const& iWhyFailed):
      product_(nullptr),
      desc_(nullptr),
      productProvenance_(nullptr),
      whyFailed_(iWhyFailed) {}
    
    ~OutputHandle() {}

    void swap(OutputHandle& other) {
      using std::swap;
      std::swap(product_, other.product_);
      std::swap(desc_, other.desc_);
      std::swap(productProvenance_, other.productProvenance_);
      swap(whyFailed_,other.whyFailed_);
    }

    
    OutputHandle& operator=(OutputHandle const& rhs) {
      OutputHandle temp(rhs);
      this->swap(temp);
      return *this;
    }

    bool isValid() const {
      return product_ && desc_ &&productProvenance_;
    }

    bool failedToGet() const {
      return 0 != whyFailed_.get();
    }
    
    WrapperBase const* wrapper() const {
      return product_;
    }

    std::shared_ptr<cms::Exception> whyFailed() const {
      return whyFailed_;
    }

    ProductProvenance const* productProvenance() const {
      return productProvenance_;
    }

    BranchDescription const* desc() const {
      return desc_;
    }

  private:
    WrapperBase const* product_;
    BranchDescription const* desc_;
    ProductProvenance* productProvenance_;
    std::shared_ptr<cms::Exception> whyFailed_;
  };

  // Free swap function
  inline
  void
  swap(OutputHandle& a, OutputHandle& b) {
    a.swap(b);
  }
}

#endif
