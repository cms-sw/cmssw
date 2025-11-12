#ifndef DataFormats_Common_interface_RefProdVector_h
#define DataFormats_Common_interface_RefProdVector_h

#include <vector>

#include "DataFormats/Common/interface/RefProd.h"

namespace edm {
  template <typename T>
  using RefProdVector = std::vector<edm::RefProd<T>>;
}  // namespace edm

#endif  // DataFormats_Common_interface_RefProdVector_h
