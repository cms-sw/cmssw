#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace {
  void throwInvalidHandleDeref() {
    throw cms::Exception("DereferenceUnsetHandle")
        << "An attempt was made to dereference an edm::Handle which was never set.";
  }
  void throwInvalidHandleProv() {
    throw cms::Exception("ProvenanceFromUnsetHandle")
        << "An attempt was made to get the ProductId from an edm::Handle which was never set.";
  }
}  // namespace

namespace edm {
  void const* HandleBase::productStorage() const {
    if UNLIKELY (not product_) {
      if LIKELY (static_cast<bool>(whyFailedFactory_)) {
        whyFailedFactory_->make()->raise();
      } else {
        throwInvalidHandleDeref();
      }
    }
    return product_;
  }

  ProductID HandleBase::id() const {
    if UNLIKELY (not prov_) {
      if LIKELY (static_cast<bool>(whyFailedFactory_)) {
        whyFailedFactory_->make()->raise();
      } else {
        throwInvalidHandleProv();
      }
    }
    return prov_->productID();
  }
}  // namespace edm
