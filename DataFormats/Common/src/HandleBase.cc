#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  void const*
  HandleBase::productStorage() const {
    if (whyFailedFactory_) {
      whyFailedFactory_->make()->raise();
    }
    return product_;
  }

  ProductID
  HandleBase::id() const {
    if (whyFailedFactory_) {
      whyFailedFactory_->make()->raise();
    }
    return prov_->productID();
  }
}
