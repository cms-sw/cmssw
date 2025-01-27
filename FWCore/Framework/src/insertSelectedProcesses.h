#ifndef FWCore_Framework_insertSelectedProcesses_h
#define FWCore_Framework_insertSelectedProcesses_h

#include <set>
#include <string>
#include "DataFormats/Provenance/interface/ProductDescriptionFwd.h"

namespace edm {

  void insertSelectedProcesses(ProductDescription const& desc,
                               std::set<std::string>& processes,
                               std::set<std::string>& processesWithKeptProcessBlockProducts);
}  // namespace edm
#endif
