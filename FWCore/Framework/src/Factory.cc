
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/interface/maker/MakerPluginFactory.h"
#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
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

  static void annotateExceptionAndRethrow(cms::Exception& except,
                                          const MakeModuleParams& p,
                                          std::string const& modtype,
                                          ModuleTypeResolverBase const* resolver) {
    if (not resolver) {
      throw except;
    }
    //if needed, create list of alternative types that were tried
    std::string alternativeTypes;
    auto index = resolver->kInitialIndex;
    auto newType = modtype;
    int tries = 0;
    do {
      ++tries;
      if (not alternativeTypes.empty()) {
        alternativeTypes.append(", ");
      }
      auto [ttype, tindex] = resolver->resolveType(std::move(newType), index);
      newType = std::move(ttype);
      index = tindex;
      alternativeTypes.append(newType);
    } while (index != resolver->kLastIndex);
    if (tries == 1 and alternativeTypes == modtype) {
      throw except;
    }
    alternativeTypes.insert(0, "These alternative types were tried: ");
    except.addAdditionalInfo(alternativeTypes);
    throw except;
  }

  Maker* Factory::findMaker(const MakeModuleParams& p, ModuleTypeResolverBase const* resolver) const {
    std::string modtype = p.pset_->getParameter<std::string>("@module_type");
    FDEBUG(1) << "Factory: module_type = " << modtype << std::endl;
    MakerMap::iterator it = makers_.find(modtype);

    if (it == makers_.end()) {
      auto make = [](auto resolver, const auto& modtype, auto const& p) {
        if (resolver) {
          auto index = resolver->kInitialIndex;
          auto newType = modtype;
          do {
            auto [ttype, tindex] = resolver->resolveType(std::move(newType), index);
            newType = std::move(ttype);
            index = tindex;
            auto m = MakerPluginFactory::get()->tryToCreate(newType);
            if (m) {
              return m;
            }
          } while (index != resolver->kLastIndex);
          try {
            //failed to find a plugin
            return MakerPluginFactory::get()->create(modtype);
          } catch (cms::Exception& iExcept) {
            annotateExceptionAndRethrow(iExcept, p, modtype, resolver);
          }
        }
        return MakerPluginFactory::get()->create(modtype);
      };
      std::unique_ptr<Maker> wm = make(resolver, modtype, p);
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
