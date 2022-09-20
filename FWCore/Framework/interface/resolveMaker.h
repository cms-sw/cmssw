#ifndef FWCore_Framework_interface_resolveMaker_h
#define FWCore_Framework_interface_resolveMaker_h

#include "FWCore/Framework/interface/ModuleTypeResolverBase.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>

namespace edm::detail {
  void annotateResolverMakerExceptionAndRethrow(cms::Exception& except,
                                                std::string const& modtype,
                                                ModuleTypeResolverBase const* resolver);

  template <typename TFactory>
  auto resolveMaker(std::string const& moduleType, ModuleTypeResolverBase const* resolver) {
    if (resolver) {
      auto index = resolver->kInitialIndex;
      auto newType = moduleType;
      do {
        auto [ttype, tindex] = resolver->resolveType(std::move(newType), index);
        newType = std::move(ttype);
        index = tindex;
        auto m = TFactory::get()->tryToCreate(newType);
        if (m) {
          return m;
        }
      } while (index != resolver->kLastIndex);
      try {
        //failed to find a plugin
        return TFactory::get()->create(moduleType);
      } catch (cms::Exception& iExcept) {
        detail::annotateResolverMakerExceptionAndRethrow(iExcept, moduleType, resolver);
      }
    }
    return TFactory::get()->create(moduleType);
  }
}  // namespace edm::detail

#endif
