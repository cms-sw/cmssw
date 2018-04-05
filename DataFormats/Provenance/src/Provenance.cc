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

  Provenance::Provenance() : Provenance{StableProvenance()} {
  }

  Provenance::Provenance(std::shared_ptr<BranchDescription const> const& p, ProductID const& pid) :
    stableProvenance_(p, pid),
    store_() {
  }

  Provenance::Provenance(StableProvenance const& stable) :
    stableProvenance_(stable),
    store_() {
  }

  ProductProvenance const*
  Provenance::productProvenance() const {
    if(!store_) {
      return nullptr;
    }
    return store_->branchIDToProvenance(originalBranchID());
  }

  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    stable().write(os);
    auto pp = productProvenance();
    if (pp != nullptr) {
      pp->write(os);
    }
  }

  bool operator==(Provenance const& a, Provenance const& b) {
    return a.stable() == b.stable();
  }


  void
  Provenance::swap(Provenance& iOther) {
    stableProvenance_.swap(iOther.stableProvenance_);
    std::swap(store_,iOther.store_);
 }
}
