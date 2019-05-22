#include "DataFormats/Provenance/interface/StableProvenance.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <algorithm>
#include <cassert>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  StableProvenance::StableProvenance() : StableProvenance{std::shared_ptr<BranchDescription const>(), ProductID()} {}

  StableProvenance::StableProvenance(std::shared_ptr<BranchDescription const> const& p, ProductID const& pid)
      : branchDescription_(p), productID_(pid), processHistory_() {}

  bool StableProvenance::getProcessConfiguration(ProcessConfiguration& pc) const {
    return processHistory_->getConfigurationForProcess(processName(), pc);
  }

  ReleaseVersion StableProvenance::releaseVersion() const {
    ProcessConfiguration pc;
    assert(getProcessConfiguration(pc));
    return pc.releaseVersion();
  }

  void StableProvenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the first pass.
    branchDescription().write(os);
  }

  bool operator==(StableProvenance const& a, StableProvenance const& b) {
    return a.branchDescription() == b.branchDescription();
  }

  void StableProvenance::swap(StableProvenance& iOther) {
    branchDescription_.swap(iOther.branchDescription_);
    productID_.swap(iOther.productID_);
    std::swap(processHistory_, iOther.processHistory_);
  }
}  // namespace edm
