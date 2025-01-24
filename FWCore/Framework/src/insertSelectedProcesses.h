#ifndef FWCore_Framework_insertSelectedProcesses_h
#define FWCore_Framework_insertSelectedProcesses_h

#include <set>
#include <string>

namespace edm {

  class ProductDescription;

  void insertSelectedProcesses(ProductDescription const& desc,
                               std::set<std::string>& processes,
                               std::set<std::string>& processesWithKeptProcessBlockProducts);
}  // namespace edm
#endif
