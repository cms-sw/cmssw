#include "PackingSetupFactory.h"

EDM_REGISTER_PLUGINFACTORY(l1t::PackingSetupFactoryT, "PackingSetupFactory");

namespace l1t {
  const PackingSetupFactory PackingSetupFactory::instance_;

  std::unique_ptr<PackingSetup> PackingSetupFactory::make(const std::string& type) const {
    return PackingSetupFactoryT::get()->create("l1t::" + type);
  }

  void PackingSetupFactory::fillDescription(edm::ParameterSetDescription& desc) const {
    for (const auto& info : PackingSetupFactoryT::get()->available()) {
      PackingSetupFactoryT::get()->create(info.name_)->fillDescription(desc);
    }
  }
}  // namespace l1t
