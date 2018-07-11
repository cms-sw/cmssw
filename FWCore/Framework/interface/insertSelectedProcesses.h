#ifndef FWCore_Framework_insertSelectedProcesses_h
#define FWCore_Framework_insertSelectedProcesses_h

#include <set>
#include <string>

namespace edm {

  class BranchDescription;

  void insertSelectedProcesses(BranchDescription const& desc,
                               std::set<std::string>& processes);
}
#endif
