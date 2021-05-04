#include "FWCore/Framework/interface/ProducesCollector.h"

namespace edm {

  ProducesCollector::ProducesCollector(ProducesCollector const& other) : helper_(get_underlying(other.helper_)) {}

  ProducesCollector& ProducesCollector::operator=(ProducesCollector const& other) {
    helper_ = get_underlying(other.helper_);
    return *this;
  }

  ProductRegistryHelper::BranchAliasSetter ProducesCollector::produces(const TypeID& id,
                                                                       std::string instanceName,
                                                                       bool recordProvenance) {
    return helper_->produces(id, std::move(instanceName), recordProvenance);
  }

  ProducesCollector::ProducesCollector(ProductRegistryHelper* helper) : helper_(helper) {}

}  // namespace edm
