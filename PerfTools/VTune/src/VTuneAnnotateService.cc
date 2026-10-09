#include "PerfTools/VTune/interface/VTuneAnnotateService.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

namespace edm {
  VTuneAnnotateService::VTuneAnnotateService(const ParameterSet& iPS, ActivityRegistry& iRegistry)
      : targetModules_(iPS.getUntrackedParameter<std::vector<std::string>>("targetModules")),
        ittDomain_(__itt_domain_create("CMSSW.ModuleTracker")) {
    iRegistry.watchPreModuleEvent(this, &VTuneAnnotateService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VTuneAnnotateService::postModuleEvent);
  }

  bool VTuneAnnotateService::isTargetModule(std::string const& label) const {
    return std::find(targetModules_.begin(), targetModules_.end(), label) != targetModules_.end();
  }

  void VTuneAnnotateService::preModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (isTargetModule(moduleLabel) || isTargetModule(moduleName) || isTargetModule("all")) {
      std::string handleName = moduleName + "/" + moduleLabel;
      __itt_string_handle* handle = __itt_string_handle_create(handleName.c_str());
      __itt_task_begin(ittDomain_, __itt_null, __itt_null, handle);
    }
  }

  void VTuneAnnotateService::postModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (isTargetModule(moduleLabel) || isTargetModule(moduleName) || isTargetModule("all")) {
      __itt_task_end(ittDomain_);
    }
  }
}  // namespace edm
