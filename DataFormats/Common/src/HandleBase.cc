#include "DataFormats/Common/interface/HandleBase.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  void const*
  HandleBase::productStorage() const {
    if (whyFailed_) {
      throw *whyFailed_;
    }
    return product_;
  }

  ProductID
  HandleBase::id() const {
    return prov_->productID();
  }
}
