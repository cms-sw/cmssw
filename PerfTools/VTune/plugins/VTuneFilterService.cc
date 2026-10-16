#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include <vector>
#include <string>
#include <algorithm>

#include "ittnotify.h"

namespace perftools {
  class VTuneFilterService {
  public:
    VTuneFilterService(const edm::ParameterSet&, edm::ActivityRegistry&);

    void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
    static void fillDescriptions(edm::ConfigurationDescriptions& desc);

  private:
    std::vector<std::string> moduleNames_;
    bool isTargetModule(std::string const& label) const;
    bool allModules_ = false;
  };

  VTuneFilterService::VTuneFilterService(const edm::ParameterSet& iPS, edm::ActivityRegistry& iRegistry)
      : moduleNames_(iPS.getUntrackedParameter<std::vector<std::string>>("moduleNames", std::vector<std::string>{})),
        allModules_(iPS.getUntrackedParameter<bool>("allModules", false)) {
    iRegistry.watchPreModuleEvent(this, &VTuneFilterService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VTuneFilterService::postModuleEvent);
  }

  bool VTuneFilterService::isTargetModule(std::string const& label) const {
    return std::find(moduleNames_.begin(), moduleNames_.end(), label) != moduleNames_.end();
  }

  void VTuneFilterService::preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (allModules_ || isTargetModule(moduleLabel) || isTargetModule(moduleName)) {
      __itt_resume();
    }
  }

  void VTuneFilterService::postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (allModules_ || isTargetModule(moduleLabel) || isTargetModule(moduleName)) {
      __itt_pause();
    }
  }

  void VTuneFilterService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<std::vector<std::string>>("moduleNames", std::vector<std::string>{})
        ->setComment("List of module labels to enable profiling for, defaults to empty");
    desc.addUntracked<bool>("allModules", false)->setComment("Enable profiling for all modules, defaults to false");
    descriptions.add("VTuneFilterService", desc);
    std::string comment =
        "VTuneFilterService enables VTune profiling for selected modules or all modules using the allModules option. "
        "It uses itt_resume/itt_pause at the start/end of the module event execution. "
        "VTune must be started paused. To profile only program execution module event processing,"
        "cmsRun must be run single threaded as itt_resume/itt_pause starts/stops "
        "profiling for the whole process.";
    descriptions.setComment(comment);
  }
}  // namespace perftools
using namespace perftools;
DEFINE_FWK_SERVICE(VTuneFilterService);
