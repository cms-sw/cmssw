#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  void const*
  HandleBase::productStorage() const {
    if (whyFailedFactory_) {
      throw *whyFailedFactory_->make();
    }
    return product_;
  }

  ProductID
  HandleBase::id() const {
    if (whyFailedFactory_) {
      throw *whyFailedFactory_->make();
    }
    return prov_->productID();
  }
}
