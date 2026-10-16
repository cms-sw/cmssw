#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"

#include <vector>
#include <string>
#include <algorithm>

#include "ittnotify.h"

namespace perftools {
  class VTuneAnnotateService {
  public:
    VTuneAnnotateService(const edm::ParameterSet&, edm::ActivityRegistry&);

    void preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
    void postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const&);
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::vector<std::string> moduleNames_;
    bool isTargetModule(std::string const& label) const;
    bool allModules_ = false;
    __itt_domain* ittDomain_;
  };

  VTuneAnnotateService::VTuneAnnotateService(const edm::ParameterSet& iPS, edm::ActivityRegistry& iRegistry)
      : moduleNames_(iPS.getUntrackedParameter<std::vector<std::string>>("moduleNames", std::vector<std::string>{})),
        allModules_(iPS.getUntrackedParameter<bool>("allModules", false)),
        ittDomain_(
            __itt_domain_create(iPS.getUntrackedParameter<std::string>("ittDomain", "CMSSW.ModuleTracker").c_str())) {
    iRegistry.watchPreModuleEvent(this, &VTuneAnnotateService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VTuneAnnotateService::postModuleEvent);
  }

  bool VTuneAnnotateService::isTargetModule(std::string const& label) const {
    return std::find(moduleNames_.begin(), moduleNames_.end(), label) != moduleNames_.end();
  }

  void VTuneAnnotateService::preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (allModules_ || isTargetModule(moduleLabel) || isTargetModule(moduleName)) {
      std::string handleName = moduleName + "/" + moduleLabel;
      __itt_string_handle* handle = __itt_string_handle_create(handleName.c_str());
      __itt_task_begin(ittDomain_, __itt_null, __itt_null, handle);
    }
  }

  void VTuneAnnotateService::postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();

    if (allModules_ || isTargetModule(moduleLabel) || isTargetModule(moduleName)) {
      __itt_task_end(ittDomain_);
    }
  }

  void VTuneAnnotateService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<std::vector<std::string>>("moduleNames", std::vector<std::string>{})
        ->setComment("List of module labels to annotate, defaults to empty");
    desc.addUntracked<bool>("allModules", false)->setComment("Annotate all modules, defaults to false");
    desc.addUntracked<std::string>("ittDomain", "CMSSW.ModuleTracker")
        ->setComment("ITT domain for the tasks, defaults to CMSSW.ModuleTracker");
    descriptions.add("VTuneAnnotateService", desc);
    std::string comment =
        "VTuneAnnotateService annotates module event execution with ITT tasks. The label of the task is constructed "
        "from the module name and module label. It uses itt_task_begin/itt_task_end at the start/end of the module "
        "event execution."
        " The ITT tasks are created and destroyed automatically at the start and end of each thread so cmsRun does not "
        "need to be run single threaded. "
        "In the VTune UI bottom-up view, these tasks will appear under the ITT domain specified by the ittDomain "
        "parameter.";
    descriptions.setComment(comment);
  }
}  // namespace perftools
using perftools::VTuneAnnotateService;
DEFINE_FWK_SERVICE(VTuneAnnotateService);
