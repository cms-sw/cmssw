#include "PerfTools/VtuneFilter/interface/VtuneAnnotateService.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

namespace edm {
  VtuneAnnotateService::VtuneAnnotateService(const ParameterSet& iPS, ActivityRegistry& iRegistry)
      : targetModules_(iPS.getUntrackedParameter<std::vector<std::string>>("targetModules")),
        ittDomain_(__itt_domain_create("CMSSW.ModuleTracker")) {
    iRegistry.watchPreModuleEvent(this, &VtuneAnnotateService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VtuneAnnotateService::postModuleEvent);
  }

  bool VtuneAnnotateService::isTargetModule(std::string const& label) const {
    return std::find(targetModules_.begin(), targetModules_.end(), label) != targetModules_.end();
  }

  void VtuneAnnotateService::preModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (isTargetModule(moduleLabel) || isTargetModule(moduleName)) {
      std::string handleName = moduleName + "/" + moduleLabel;
      __itt_string_handle* handle = __itt_string_handle_create(handleName.c_str());
      __itt_task_begin(ittDomain_, __itt_null, __itt_null, handle);
    }
  }

  void VtuneAnnotateService::postModuleEvent(StreamContext const&, ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (isTargetModule(moduleLabel) || isTargetModule(moduleName)) {
      __itt_task_end(ittDomain_);
    }
  }
}  // namespace edm
