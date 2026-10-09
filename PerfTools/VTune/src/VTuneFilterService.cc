#include "PerfTools/VTune/interface/VTuneFilterService.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

namespace edm {
  VTuneFilterService::VTuneFilterService(const ParameterSet& iPS, ActivityRegistry& iRegistry)
      : targetModules_(iPS.getUntrackedParameter<std::vector<std::string>>("targetModules")) {
    iRegistry.watchPreModuleEvent(this, &VTuneFilterService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VTuneFilterService::postModuleEvent);
  }

  bool VTuneFilterService::isTargetModule(std::string const& label) const {
    return std::find(targetModules_.begin(), targetModules_.end(), label) != targetModules_.end();
  }

  void VTuneFilterService::preModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (isTargetModule(moduleLabel) || isTargetModule(moduleName) || isTargetModule("all")) {
      __itt_resume();
    }
  }

  void VTuneFilterService::postModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (isTargetModule(moduleLabel) || isTargetModule(moduleName) || isTargetModule("all")) {
      __itt_pause();
    }
  }
}  // namespace edm
