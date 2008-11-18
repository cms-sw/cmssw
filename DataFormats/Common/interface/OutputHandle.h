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

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/shared_ptr.hpp>

namespace edm {
  class EDProduct;
  template <typename T>
  class OutputHandle {
  public:
    OutputHandle() :
      wrap_(0),
      desc_(0),
      entryInfo_() {}

    OutputHandle(OutputHandle const& h) :
      wrap_(h.wrap_),
      desc_(h.desc_),
      entryInfo_(h.entryInfo_),
      whyFailed_(h.whyFailed_){}

    OutputHandle(EDProduct const* prod, ConstBranchDescription const* desc, boost::shared_ptr<EventEntryInfo> entryInfo) :
      wrap_(prod),
      desc_(desc),
      entryInfo_(boost::shared_ptr<T>(new T(*entryInfo))) {}

    ///Used when the attempt to get the data failed
    OutputHandle(const boost::shared_ptr<cms::Exception>& iWhyFailed):
      wrap_(0),
      desc_(0),
      entryInfo_(),
      whyFailed_(iWhyFailed) {}
    
    ~OutputHandle() {}

    void swap(OutputHandle& other) {
      using std::swap;
      std::swap(wrap_, other.wrap_);
      std::swap(desc_, other.desc_);
      std::swap(entryInfo_, other.entryInfo_);
      swap(whyFailed_,other.whyFailed_);
    }

    
    OutputHandle& operator=(OutputHandle const& rhs) {
      OutputHandle temp(rhs);
      this->swap(temp);
      return *this;
    }

    bool isValid() const {
      return wrap_ && desc_ &&entryInfo_;
    }

    bool failedToGet() const {
      return 0 != whyFailed_.get();
    }
    
    EDProduct const* wrapper() const {
      return wrap_;
    }

    ProductID id() const {
      if (!entryInfo_) {
        return ProductID();
      }
      return entryInfo_->productID();
    }

    boost::shared_ptr<cms::Exception> whyFailed() const {
      return whyFailed_;
    }

    T const* entryInfo() const {
      return entryInfo_.get();
    }

    boost::shared_ptr<T> entryInfoSharedPtr() const {
      return entryInfo_;
    }

    ConstBranchDescription const* desc() const {
      return desc_;
    }

  private:
    EDProduct const* wrap_;
    ConstBranchDescription const* desc_;
    boost::shared_ptr<T> entryInfo_;
    boost::shared_ptr<cms::Exception> whyFailed_;
  };

  // Free swap function
  template <typename T>
  inline
  void
  swap(OutputHandle<T>& a, OutputHandle<T>& b) {
    a.swap(b);
  }
}

#endif
