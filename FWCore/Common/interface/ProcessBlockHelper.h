#ifndef FWCore_Common_ProcessBlockHelper_h
#define FWCore_Common_ProcessBlockHelper_h

/** \class edm::ProcessBlockHelper

\author W. David Dagenhart, created 15 September, 2020

*/

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"

#include <set>
#include <string>
#include <vector>

namespace edm {

  class ProcessBlockHelper {
  public:
    std::vector<std::string> const& processesWithProcessBlockProducts() const {
      return processesWithProcessBlockProducts_;
    }

    void initializeFromPrimaryInput(ProductRegistry const&, StoredProcessBlockHelper const&);
    void updateForNewProcess(ProductRegistry const&, std::string const& processName);
    void updateFromParentProcess(ProcessBlockHelper const& parentProcessBlockHelper, ProductRegistry const&);
    void updateAfterProductSelection(std::set<std::string> const& processesWithKeptProcessBlockProducts,
                                     ProcessBlockHelper const&);

  private:
    // Includes processes with ProcessBlock branches present
    // in the first input file. At each processing step the
    // new process will be added at the end if there are non-transient
    // ProcessBlock products being produced. Output modules
    // will write a copy of this to persistent storage after
    // removing any process without at least one kept
    // ProcessBlock branch.
    std::vector<std::string> processesWithProcessBlockProducts_;

    // TODO for now this is just a placeholder...

    // Events/Runs/Lumis hold an index into the outer vector.
    // The elements of the inner vector correspond to the
    // processes in processesWithPBProducts_ (1 to 1 correspondence
    // and in the same order).
    std::vector<std::vector<unsigned int>> outputTTreeEntries_;

    bool initializedFromInput_ = false;
  };
}  // namespace edm
#endif
