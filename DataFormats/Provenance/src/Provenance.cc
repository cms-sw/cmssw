#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProductProvenanceRetriever.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <algorithm>
#include <cassert>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  Provenance::Provenance() : Provenance{std::shared_ptr<BranchDescription const>(), ProductID()} {
  }

  Provenance::Provenance(std::shared_ptr<BranchDescription const> const& p, ProductID const& pid) :
    branchDescription_(p),
    productID_(pid),
    processHistory_(),
    store_() {
  }

  ProductProvenance const*
  Provenance::productProvenance() const {
    if(!store_) {
      return nullptr;
    }
    return store_->branchIDToProvenance(branchDescription_->branchID());
  }

  bool
  Provenance::getProcessConfiguration(ProcessConfiguration& pc) const {
    return processHistory_->getConfigurationForProcess(processName(), pc);
  }

  ReleaseVersion
  Provenance::releaseVersion() const {
    ProcessConfiguration pc;
    assert(getProcessConfiguration(pc));
    return pc.releaseVersion();
  }

  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    product().write(os);
    auto pp = productProvenance();
    if (pp != nullptr) {
      pp->write(os);
    }
  }

  bool operator==(Provenance const& a, Provenance const& b) {
    return a.product() == b.product();
  }


  void
  Provenance::swap(Provenance& iOther) {
    branchDescription_.swap(iOther.branchDescription_);
    productID_.swap(iOther.productID_);
    std::swap(processHistory_, iOther.processHistory_);
    std::swap(store_,iOther.store_);
 }
}
