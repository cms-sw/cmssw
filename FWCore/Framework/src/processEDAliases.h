#ifndef FWCore_Framework_src_processEDAliases_h
#define FWCore_Framework_src_processEDAliases_h

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <unordered_set>
#include <string>
#include <vector>

namespace edm::detail {
  /**
   * Process and insert EDAliases to ProductRegistry
   *
   * Processes only those EDAliases whose names are given in
   * aliasNamesToProcess. If aliasModulesToProcess is not empty, only
   * those alias branches that point to modules named in
   * aliasModulesToProcess are processed.
   */
  void processEDAliases(std::vector<std::string> const& aliasNamesToProcess,
                        std::unordered_set<std::string> const& aliasModulesToProcess,
                        ParameterSet const& proc_pset,
                        std::string const& processName,
                        ProductRegistry& preg);
}  // namespace edm::detail

#endif
