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
    bool isTargetModule(unsigned int id) const;

    std::vector<std::string> moduleNames_;
    std::vector<unsigned int> moduleIDs_;
    bool allModules_ = false;
  };

  VTuneFilterService::VTuneFilterService(const edm::ParameterSet& iPS, edm::ActivityRegistry& iRegistry)
      : moduleNames_(iPS.getUntrackedParameter<std::vector<std::string>>("moduleNames")),
        allModules_(iPS.getUntrackedParameter<bool>("allModules")) {
    iRegistry.watchPreModuleEvent(this, &VTuneFilterService::preModuleEvent);
    iRegistry.watchPostModuleEvent(this, &VTuneFilterService::postModuleEvent);
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

  bool VTuneFilterService::isTargetModule(unsigned int id) const {
    return allModules_ or std::binary_search(moduleIDs_.begin(), moduleIDs_.end(), id);
  }

  void VTuneFilterService::preModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    if (isTargetModule(mcc.moduleDescription()->id())) {
      __itt_resume();
    }
  }

  void VTuneFilterService::postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
    if (isTargetModule(mcc.moduleDescription()->id())) {
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
