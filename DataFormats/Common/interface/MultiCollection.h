#ifndef DataFormats_Common_interface_MultiCollection_h
#define DataFormats_Common_interface_MultiCollection_h

#include <vector>

#include "DataFormats/Common/interface/RefProd.h"

namespace edm {
  template <typename T>
  using MultiCollection = std::vector<edm::RefProd<T>>;
}  // namespace edm

#endif  // DataFormats_Common_interface_MultiCollection_h
