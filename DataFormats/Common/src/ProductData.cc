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

  ProductData::ProductData(boost::shared_ptr<ConstBranchDescription> bd) :
    wrapper_(),
    prov_(bd, ProductID()) {
  }

  // For use by FWLite
  ProductData::ProductData(void const* product, Provenance const& prov) :
    wrapper_(product, do_nothing_deleter()),
    prov_(prov) {
  }

  void
  ProductData::resetBranchDescription(boost::shared_ptr<ConstBranchDescription> bd) {
    prov_.setBranchDescription(bd);
  }
}
