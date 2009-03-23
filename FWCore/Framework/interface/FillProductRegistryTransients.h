#ifndef FWCore_Framework_fillProductRegistryTransients_h
#define FWCore_Framework_fillProductRegistryTransients_h

#include <vector>

namespace edm {
  class ProductRegistry;
  class ProcessConfiguration;
  void
  fillProductRegistryTransients(std::vector<ProcessConfiguration> const& pcVec, ProductRegistry& preg);
}
#endif
