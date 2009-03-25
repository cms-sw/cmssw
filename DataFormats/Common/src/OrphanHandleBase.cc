#include "DataFormats/Common/interface/OrphanHandleBase.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace edm {
  void const*
  OrphanHandleBase::productStorage() const {
    return product_;
  }

  ProductID
  OrphanHandleBase::id() const {
    return id_;
  }
}
