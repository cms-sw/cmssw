#ifndef FWCore_ParameterSet_FillProductRegistryTransients_h
#define FWCore_ParameterSet_FillProductRegistryTransients_h

#include <vector>

// fillProductRegistry()
// This free function reads information from the process parameter set
// and writes information derived from this into the ProductRegistry.
// It really does not belong in ParameterSet, but ParameterSet is the only existing
// package in which it can go without introducing additional package dependencies.

namespace edm {
  class ProductRegistry;
  class ProcessConfiguration;
  void
  fillProductRegistryTransients(std::vector<ProcessConfiguration> const& pcVec, ProductRegistry& preg, bool okToRegister = false);
}
#endif
