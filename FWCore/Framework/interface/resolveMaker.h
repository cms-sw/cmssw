#ifndef FWCore_Framework_interface_resolveMaker_h
#define FWCore_Framework_interface_resolveMaker_h

#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
#include "FWCore/Framework/interface/ModuleTypeResolverMaker.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>

namespace edm::detail {
  void annotateResolverMakerExceptionAndRethrow(cms::Exception& except,
                                                std::string const& modtype,
                                                ModuleTypeResolverBase const* resolver);

  // Returns a non-owning pointer to the maker. Can be nullptr if
  // failed to insert the maker to the cache
  template <typename TFactory, typename TCache>
  auto resolveMaker(std::string const& moduleType,
                    ModuleTypeResolverMaker const* resolverMaker,
                    edm::ParameterSet const& modulePSet,
                    TCache& makerCache) -> typename TCache::mapped_type::element_type const* {
    if (resolverMaker) {
      auto resolver = resolverMaker->makeResolver(modulePSet);
      auto index = resolver->kInitialIndex;
      auto newType = moduleType;
      do {
        auto [ttype, tindex] = resolver->resolveType(std::move(newType), index);
        newType = std::move(ttype);
        index = tindex;
        // try the maker cache first
        auto found = makerCache.find(newType);
        if (found != makerCache.end()) {
          return found->second.get();
        }

        // if not in cache, then try to create
        auto m = TFactory::get()->tryToCreate(newType);
        if (m) {
          //FDEBUG(1) << "Factory:  created worker of type " << newType << std::endl;
          auto [it, succeeded] = makerCache.emplace(newType, std::move(m));
          if (not succeeded) {
            return nullptr;
          }
          return it->second.get();
        }
        // not found, try next one
      } while (index != resolver->kLastIndex);
      try {
        //failed to find a plugin
        auto m = TFactory::get()->create(moduleType);
        return nullptr;  // dummy return, the create() call throws an exception
      } catch (cms::Exception& iExcept) {
        detail::annotateResolverMakerExceptionAndRethrow(iExcept, moduleType, resolver.get());
      }
    }
    auto [it, succeeded] = makerCache.emplace(moduleType, TFactory::get()->create(moduleType));
    //FDEBUG(1) << "Factory:  created worker of type " << moduleType << std::endl;
    if (not succeeded) {
      return nullptr;
    }
    return it->second.get();
  }
}  // namespace edm::detail

#endif
