#include "PerfTools/VtuneFilter/interface/VtuneAnnotateService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

namespace edm {
  VtuneAnnotateService::VtuneAnnotateService(const ParameterSet& iPS, ActivityRegistry& iRegistry) {
    targetModules_ = iPS.getUntrackedParameter<std::vector<std::string>>("targetModules");

    // Create a global ITT domain for your CMSSW tracking
    ittDomain_ = __itt_domain_create("CMSSW.ModuleTracker");

    iRegistry.watchPreModuleEvent(this, &VtuneAnnotateService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VtuneAnnotateService::postModuleEvent);
  }

  bool edm::VtuneAnnotateService::isTargetModule(std::string const& label) const {
    return std::find(targetModules_.begin(), targetModules_.end(), label) != targetModules_.end();
  }

  void VtuneAnnotateService::preModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();

    // If it's a module we want to IGNORE/DISABLE profiling for:
    if (isTargetModule(moduleLabel)) {
      // Dynamically generate a string handle for this specific module name
      __itt_string_handle* handle = __itt_string_handle_create(moduleLabel.c_str());

      // Start a task on THIS thread only. Global profiling remains ACTIVE.
      __itt_task_begin(ittDomain_, __itt_null, __itt_null, handle);
    }
  }

  void VtuneAnnotateService::postModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();

    if (isTargetModule(moduleLabel)) {
      // End the task on THIS thread
      __itt_task_end(ittDomain_);
    }
  }
}  // namespace edm

using edm::VtuneAnnotateService;
DEFINE_FWK_SERVICE(VtuneAnnotateService);
