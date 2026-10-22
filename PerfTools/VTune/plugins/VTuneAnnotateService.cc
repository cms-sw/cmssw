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
    void preModuleConstruction(const edm::ModuleDescription& description);
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    std::vector<std::string> moduleNames_;
    std::vector<int> moduleIDs_;
    bool isTargetModule(unsigned int id) const;
    bool allModules_ = false;
    __itt_domain* ittDomain_;
  };

  VTuneAnnotateService::VTuneAnnotateService(const edm::ParameterSet& iPS, edm::ActivityRegistry& iRegistry)
      : moduleNames_(iPS.getUntrackedParameter<std::vector<std::string>>("moduleNames")),
        allModules_(iPS.getUntrackedParameter<bool>("allModules")),
        ittDomain_(__itt_domain_create(iPS.getUntrackedParameter<std::string>("ittDomain").c_str())) {
    iRegistry.watchPreModuleEvent(this, &VTuneAnnotateService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VTuneAnnotateService::postModuleEvent);

    if (not moduleNames_.empty()) {
      iRegistry.watchPreModuleConstruction([this](auto const& description) {
        unsigned int idlabel = description.id();
        auto foundlabel = std::find(moduleNames_.begin(), moduleNames_.end(), description.moduleLabel());
        if (foundlabel != moduleNames_.end()) {
          moduleIDs_.push_back(idlabel);
        }
        auto foundname = std::find(moduleNames_.begin(), moduleNames_.end(), description.moduleName());
        if (foundname != moduleNames_.end() and foundlabel == moduleNames_.end()) {
          moduleIDs_.push_back(description.id());
        }
        std::sort(moduleIDs_.begin(), moduleIDs_.end());
      });
    }
  }

  bool VTuneAnnotateService::isTargetModule(unsigned int id) const {
    return allModules_ or std::binary_search(moduleIDs_.begin(), moduleIDs_.end(), id);
  }

  void VTuneAnnotateService::preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    std::string const& moduleLabel = mcc.moduleDescription()->moduleLabel();
    std::string const& moduleName = mcc.moduleDescription()->moduleName();
    if (isTargetModule(mcc.moduleDescription()->id())) {
      std::string handleName = moduleName + "/" + moduleLabel;
      __itt_string_handle* handle = __itt_string_handle_create(handleName.c_str());
      __itt_task_begin(ittDomain_, __itt_null, __itt_null, handle);
    }
  }

  void VTuneAnnotateService::postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    if (isTargetModule(mcc.moduleDescription()->id())) {
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
