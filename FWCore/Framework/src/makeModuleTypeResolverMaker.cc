#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
#include "FWCore/Framework/interface/makeModuleTypeResolverMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edm {
  std::unique_ptr<edm::ModuleTypeResolverMaker const> makeModuleTypeResolverMaker(edm::ParameterSet const& pset) {
    auto const& name = pset.getUntrackedParameter<std::string>("@module_type_resolver");
    if (name.empty()) {
      return nullptr;
    }
    return edm::ModuleTypeResolverMakerFactory::get()->create(name);
  }
}  // namespace edm
