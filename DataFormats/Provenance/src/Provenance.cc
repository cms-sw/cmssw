#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"

#include <algorithm>

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  Provenance::Provenance() : Provenance{boost::shared_ptr<BranchDescription const>(), ProductID()} {
  }

  Provenance::Provenance(boost::shared_ptr<BranchDescription const> const& p, ProductID const& pid) :
    branchDescription_(p),
    productID_(pid),
    processHistoryID_(),
    productProvenanceValid_(false),
    productProvenancePtr_(new ProductProvenance),
    store_() {
  }

  ProductProvenance*
  Provenance::resolve() const {
    if(!store_) {
      return 0;
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

  ProcessConfigurationID
  Provenance::processConfigurationID() const {
    if (parameterSetIDs().size() == 1) {
      return parameterSetIDs().begin()->first;
    }
    if (moduleNames().size() == 1) {
      return moduleNames().begin()->first;
    }
    // Get the ProcessHistory for this event.
    ProcessHistoryRegistry* phr = ProcessHistoryRegistry::instance();
    ProcessHistory ph;
    if (!phr->getMapped(processHistoryID(), ph)) {
      return ProcessConfigurationID();
    }

    ProcessConfiguration config;
    if (!ph.getConfigurationForProcess(processName(), config)) {
      return ProcessConfigurationID();
    }
    return config.id();
  }

  bool
  Provenance::getProcessConfiguration(ProcessConfiguration& pc) const {
    // Get the ProcessHistory for this event.
    ProcessHistoryRegistry* phr = ProcessHistoryRegistry::instance();
    ProcessHistory ph;
    if (!phr->getMapped(processHistoryID(), ph)) {
      return false;
    }
    return ph.getConfigurationForProcess(processName(), pc);
  }

  ReleaseVersion
  Provenance::releaseVersion() const {
    ProcessConfiguration pc;
    assert(getProcessConfiguration(pc));
    return pc.releaseVersion();
  }

  ParameterSetID
  Provenance::psetID() const {
    if (product().parameterSetID().isValid()) {
      return product().parameterSetID();
    }
    if (parameterSetIDs().size() == 1) {
      return parameterSetIDs().begin()->second;
    }
    std::map<ProcessConfigurationID, ParameterSetID>::const_iterator it =
        parameterSetIDs().find(processConfigurationID());
    if (it == parameterSetIDs().end()) {
      return ParameterSetID();
    }
    return it->second;
  }

  std::string
  Provenance::moduleName() const {
    if (!product().moduleName().empty()) {
      return product().moduleName();
    }
    if (moduleNames().size() == 1) {
      return moduleNames().begin()->second;
    }
    std::map<ProcessConfigurationID, std::string>::const_iterator it =
        moduleNames().find(processConfigurationID());
    if (it == moduleNames().end()) {
      return std::string();
    }
    return it->second;
  }

  void
  Provenance::write(std::ostream& os) const {
    // This is grossly inadequate, but it is not critical for the
    // first pass.
    product().write(os);
    ProductProvenance* pp = productProvenance();
    if (pp != 0) {
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
    std::swap(processHistoryID_, iOther.processHistoryID_);
    std::swap(productProvenanceValid_, iOther.productProvenanceValid_);
    productProvenancePtr_.swap(iOther.productProvenancePtr_);
    store_.swap(iOther.store_);
 }
}
