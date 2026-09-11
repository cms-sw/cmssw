#include "PerfTools/VtuneFilter/interface/VtuneFilterService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  VtuneFilterService::VtuneFilterService(const ParameterSet& iPS, ActivityRegistry& iRegistry) {
    targetModules_ = iPS.getUntrackedParameter<std::vector<std::string>>("targetModules");

    iRegistry.watchPreModuleEvent(this, &VtuneFilterService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VtuneFilterService::postModuleEvent);
  }

  bool edm::VtuneFilterService::isTargetModule(std::string const& label) const {
    return std::find(targetModules_.begin(), targetModules_.end(), label) != targetModules_.end();
  }

  void VtuneFilterService::preModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();

    if (isTargetModule(moduleLabel)) {
      // Tells VTune to pause collecting data for entire process
      __itt_pause();
    }
  }

  void VtuneFilterService::postModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();

    if (isTargetModule(moduleLabel)) {
      // Tells VTune to resume collecting data for entire process
      __itt_resume();
    }
  }
}  // namespace edm

using edm::VtuneFilterService;
DEFINE_FWK_SERVICE(VtuneFilterService);
