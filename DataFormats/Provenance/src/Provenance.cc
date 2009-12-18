#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProcessHistoryRegistry.h"

/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

namespace edm {

  Provenance::Provenance(BranchDescription const& p, ProductID const& pid) :
    branchDescription_(p),
    productID_(pid),
    productProvenancePtr_() {
  }

  Provenance::Provenance(ConstBranchDescription const& p, ProductID const& pid) :
    branchDescription_(p),
    productID_(pid),
    productProvenancePtr_() {
  }

  Provenance::Provenance(BranchDescription const& p, ProductID const& pid,
      boost::shared_ptr<ProductProvenance> ppr) :
    branchDescription_(p),
    productID_(pid),
    productProvenancePtr_(ppr)
  { }

  Provenance::Provenance(ConstBranchDescription const& p, ProductID const& pid,
      boost::shared_ptr<ProductProvenance> ppr) :
    branchDescription_(p),
    productID_(pid),
    productProvenancePtr_(ppr)
  { }

  boost::shared_ptr<ProductProvenance>
  Provenance::resolve () const {
    if (productProvenancePtr_.get() == 0) {
      productProvenancePtr_ = store_->branchIDToProvenance(branchDescription_.branchID());
    }
    return productProvenancePtr_;
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
    if (!phr->getMapped(store_->processHistoryID(), ph)) {
      return ProcessConfigurationID();
    }

    ProcessConfiguration config;
    if (!ph.getConfigurationForProcess(processName(), config)) {
      return ProcessConfigurationID();
    } 
    return config.id();
  }

  ReleaseVersion const&
  Provenance::releaseVersion() const {
    ProcessConfiguration pc;
    ProcessConfigurationRegistry::instance()->getMapped(processConfigurationID(), pc);
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
    productProvenance().write(os);
  }


  bool operator==(Provenance const& a, Provenance const& b) {
    return a.product() == b.product();
  }


}

