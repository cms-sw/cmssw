/*----------------------------------------------------------------------
----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/ProductData.h"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"

namespace edm {
  ProductData::ProductData() :
    wrapper_(),
    prov_() {
  }

  ProductData::ProductData(std::shared_ptr<BranchDescription const> bd) :
    wrapper_(),
    prov_(bd, ProductID()) {
  }

  // For use by FWLite
  ProductData::ProductData(WrapperBase* product, Provenance const& prov) :
    wrapper_(product, do_nothing_deleter()),
    prov_(prov) {
  }

  void
  ProductData::resetBranchDescription(std::shared_ptr<BranchDescription const> bd) {
    prov_.setBranchDescription(bd);
  }
}
