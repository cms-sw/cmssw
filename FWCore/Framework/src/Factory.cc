
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/maker/MakerPluginFactory.h"
#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
#include "FWCore/Framework/interface/resolveMaker.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::MakerPluginFactory, "CMS EDM Framework Module");
namespace edm {

  static void cleanup(const Factory::MakerMap::value_type& v) { delete v.second.get(); }

  Factory const Factory::singleInstance_;

  Factory::~Factory() { for_all(makers_, cleanup); }

  Factory::Factory()
      : makers_()

  {}

  Factory const* Factory::get() { return &singleInstance_; }

  Maker* Factory::findMaker(const MakeModuleParams& p, ModuleTypeResolverBase const* resolver) const {
    std::string modtype = p.pset_->getParameter<std::string>("@module_type");
    FDEBUG(1) << "Factory: module_type = " << modtype << std::endl;
    MakerMap::iterator it = makers_.find(modtype);

    if (it == makers_.end()) {
      std::unique_ptr<Maker> wm = detail::resolveMaker<MakerPluginFactory>(modtype, resolver);
      FDEBUG(1) << "Factory:  created worker of type " << modtype << std::endl;

      std::pair<MakerMap::iterator, bool> ret = makers_.insert(std::pair<std::string, Maker*>(modtype, wm.get()));

      it = ret.first;
      wm.release();
    }
    return it->second;
  }

  std::shared_ptr<maker::ModuleHolder> Factory::makeModule(
      const MakeModuleParams& p,
      const ModuleTypeResolverBase* resolver,
      signalslot::Signal<void(const ModuleDescription&)>& pre,
      signalslot::Signal<void(const ModuleDescription&)>& post) const {
    auto maker = findMaker(p, resolver);
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
