#include "DataFormats/Common/interface/OrphanHandleBase.h"

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
