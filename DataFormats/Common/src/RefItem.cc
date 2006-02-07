#include "DataFormats/Common/interface/RefItem.h"

namespace edm {
  void const *
  RefItem::setPtr(void const* p) const {
    return(ptr_ = p);
  }
}
