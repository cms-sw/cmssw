#include "FWCore/Framework/interface/ModuleTypeResolverMaker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include <cassert>
#include <cstring>
#include <string>
#include <unordered_map>

class ModuleTypeResolverAlpaka : public edm::ModuleTypeResolverBase {
public:
  ModuleTypeResolverAlpaka(std::string backendPrefix) : backendPrefix_(std::move(backendPrefix)) {}

  std::pair<std::string, int> resolveType(std::string basename, int index) const final {
    assert(index == kInitialIndex);
    constexpr auto kAlpaka = "@alpaka";
    auto found = basename.find(kAlpaka);
    if (found != std::string::npos) {
      if (backendPrefix_.empty()) {
        throw edm::Exception(edm::errors::LogicError)
            << "AlpakaModuleTypeResolver encountered a module with type name " << basename
            << " but the backend prefix was empty. This should not happen. Please contact framework developers.";
      }
      basename.replace(found, std::strlen(kAlpaka), "");
      basename = backendPrefix_ + basename;
    }
    return {basename, kLastIndex};
  }

private:
  std::string const backendPrefix_;
};

class ModuleTypeResolverMakerAlpaka : public edm::ModuleTypeResolverMaker {
public:
  ModuleTypeResolverMakerAlpaka() {}

  std::shared_ptr<edm::ModuleTypeResolverBase const> makeResolver(edm::ParameterSet const& modulePSet) const final {
    std::string prefix = "";
    if (modulePSet.existsAs<edm::ParameterSet>("alpaka", false)) {
      auto const& backend =
          modulePSet.getUntrackedParameter<edm::ParameterSet>("alpaka").getUntrackedParameter<std::string>("backend");
      prefix = fmt::format("alpaka_{}::", backend);

      LogDebug("AlpakaModuleTypeResolver")
          .format("AlpakaModuleTypeResolver: module {} backend prefix {}",
                  modulePSet.getParameter<std::string>("@module_label"),
                  prefix);
    }
    auto found = cache_.find(prefix);
    if (found == cache_.end()) {
      bool inserted;
      std::tie(found, inserted) = cache_.emplace(prefix, std::make_shared<ModuleTypeResolverAlpaka>(prefix));
    }
    return found->second;
  }

private:
  // no protection needed because this object is used only in single-thread context
  CMS_SA_ALLOW mutable std::unordered_map<std::string, std::shared_ptr<ModuleTypeResolverAlpaka const>> cache_;
};

#include "FWCore/Framework/interface/ModuleTypeResolverMakerFactory.h"
DEFINE_EDM_PLUGIN(edm::ModuleTypeResolverMakerFactory, ModuleTypeResolverMakerAlpaka, "ModuleTypeResolverAlpaka");
