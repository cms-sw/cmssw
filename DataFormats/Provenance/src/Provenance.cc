#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"

#include <algorithm>
#include <cassert>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  Provenance::Provenance() : Provenance{boost::shared_ptr<BranchDescription const>(), ProductID()} {
  }

  Provenance::Provenance(boost::shared_ptr<BranchDescription const> const& p, ProductID const& pid) :
    branchDescription_(p),
    productID_(pid),
    processHistory_(),
    productProvenanceValid_(false),
    productProvenancePtr_(new ProductProvenance),
    store_() {
  }

  ProductProvenance*
  Provenance::resolve() const {
    if(!store_) {
      return nullptr;
    }
    if (!productProvenanceValid_) {
      ProductProvenance const* prov  = store_->branchIDToProvenance(branchDescription_->branchID());
      if (prov) {
        *productProvenancePtr_ = *prov;
        productProvenanceValid_ = true;
      }
    }
    return productProvenancePtr_.get();
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
    ProductProvenance* pp = productProvenance();
    if (pp != nullptr) {
      pp->write(os);
    }
  }

  bool operator==(Provenance const& a, Provenance const& b) {
    return a.product() == b.product();
  }

  void
  Provenance::resetProductProvenance() const {
    *productProvenancePtr_ = ProductProvenance();
    productProvenanceValid_ = false;
  }

  void
  Provenance::setProductProvenance(ProductProvenance const& prov) const {
    *productProvenancePtr_ = prov;
    productProvenanceValid_ = true;
  }

  void
  Provenance::swap(Provenance& iOther) {
    branchDescription_.swap(iOther.branchDescription_);
    productID_.swap(iOther.productID_);
    std::swap(processHistory_, iOther.processHistory_);
    std::swap(productProvenanceValid_, iOther.productProvenanceValid_);
    productProvenancePtr_.swap(iOther.productProvenancePtr_);
    store_.swap(iOther.store_);
 }
}
