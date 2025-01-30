#include "DataFormats/Provenance/interface/StableProvenance.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"

#include <algorithm>
#include <cassert>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  StableProvenance::StableProvenance() : StableProvenance{std::shared_ptr<ProductDescription const>(), ProductID()} {}

  StableProvenance::StableProvenance(std::shared_ptr<ProductDescription const> const& p, ProductID const& pid)
      : productDescription_(p), productID_(pid) {}

  void StableProvenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the first pass.
    productDescription().write(os);
  }

  bool operator==(StableProvenance const& a, StableProvenance const& b) {
    return a.productDescription() == b.productDescription();
  }

  void StableProvenance::swap(StableProvenance& iOther) {
    productDescription_.swap(iOther.productDescription_);
    productID_.swap(iOther.productID_);
  }
}  // namespace edm
