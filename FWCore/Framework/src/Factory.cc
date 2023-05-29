
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/maker/MakerPluginFactory.h"
#include "FWCore/Framework/interface/ModuleTypeResolverMaker.h"
#include "FWCore/Framework/interface/resolveMaker.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::MakerPluginFactory, "CMS EDM Framework Module");
namespace edm {

  Factory const Factory::singleInstance_;

  Factory::~Factory() = default;

  Factory::Factory() = default;

  Factory const* Factory::get() { return &singleInstance_; }

  Maker const* Factory::findMaker(const MakeModuleParams& p, ModuleTypeResolverMaker const* resolverMaker) const {
    std::string modtype = p.pset_->getParameter<std::string>("@module_type");
    FDEBUG(1) << "Factory: module_type = " << modtype << std::endl;
    MakerMap::iterator it = makers_.find(modtype);
    if (it != makers_.end()) {
      return it->second.get();
    }
    return detail::resolveMaker<MakerPluginFactory>(modtype, resolverMaker, *p.pset_, makers_);
  }

  std::shared_ptr<maker::ModuleHolder> Factory::makeModule(
      const MakeModuleParams& p,
      const ModuleTypeResolverMaker* resolverMaker,
      signalslot::Signal<void(const ModuleDescription&)>& pre,
      signalslot::Signal<void(const ModuleDescription&)>& post) const {
    auto maker = findMaker(p, resolverMaker);
    auto mod(maker->makeModule(p, pre, post));
    return mod;
  }

  std::shared_ptr<maker::ModuleHolder> Factory::makeReplacementModule(const edm::ParameterSet& p) const {
    std::string modtype = p.getParameter<std::string>("@module_type");
    MakerMap::iterator it = makers_.find(modtype);
    if (it != makers_.end()) {
      return it->second->makeReplacementModule(p);
    }
    return std::shared_ptr<maker::ModuleHolder>{};
  }
}  // namespace edm
