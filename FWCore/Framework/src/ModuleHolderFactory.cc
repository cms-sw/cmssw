
#include "FWCore/Framework/src/ModuleHolderFactory.h"
#include "FWCore/Framework/interface/maker/ModuleMakerPluginFactory.h"
#include "FWCore/Framework/interface/ModuleTypeResolverMaker.h"
#include "FWCore/Framework/interface/resolveMaker.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::ModuleMakerPluginFactory, "CMS EDM Framework Module");
namespace edm {

  ModuleHolderFactory const ModuleHolderFactory::singleInstance_;

  ModuleHolderFactory::~ModuleHolderFactory() = default;

  ModuleHolderFactory::ModuleHolderFactory() = default;

  ModuleHolderFactory const* ModuleHolderFactory::get() { return &singleInstance_; }

  ModuleMakerBase const* ModuleHolderFactory::findMaker(const MakeModuleParams& p,
                                                        ModuleTypeResolverMaker const* resolverMaker) const {
    std::string modtype = p.pset_->getParameter<std::string>("@module_type");
    MakerMap::iterator it = makers_.find(modtype);
    if (it != makers_.end()) {
      return it->second.get();
    }
    return detail::resolveMaker<ModuleMakerPluginFactory>(modtype, resolverMaker, *p.pset_, makers_);
  }

  std::shared_ptr<maker::ModuleHolder> ModuleHolderFactory::makeModule(
      const MakeModuleParams& p,
      const ModuleTypeResolverMaker* resolverMaker,
      signalslot::Signal<void(const ModuleDescription&)>& pre,
      signalslot::Signal<void(const ModuleDescription&)>& post) const {
    auto maker = findMaker(p, resolverMaker);
    auto mod(maker->makeModule(p, pre, post));
    return mod;
  }

  std::shared_ptr<maker::ModuleHolder> ModuleHolderFactory::makeReplacementModule(const edm::ParameterSet& p) const {
    std::string modtype = p.getParameter<std::string>("@module_type");
    MakerMap::iterator it = makers_.find(modtype);
    if (it != makers_.end()) {
      return it->second->makeReplacementModule(p);
    }
    return std::shared_ptr<maker::ModuleHolder>{};
  }
}  // namespace edm
