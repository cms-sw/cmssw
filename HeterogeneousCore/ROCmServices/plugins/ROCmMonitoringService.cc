#include <iostream>

#include <hip/hip_runtime.h>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"

namespace edm {
  class StreamContext;
}

class ROCmMonitoringService {
public:
  ROCmMonitoringService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);
  ~ROCmMonitoringService() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void postModuleConstruction(edm::ModuleDescription const& desc);
  void postModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const& mcc);
  void postModuleEvent(edm::StreamContext const& sc, edm::ModuleCallingContext const& mcc);
  void postEvent(edm::StreamContext const& sc);

private:
  int numberOfDevices_ = 0;
};

ROCmMonitoringService::ROCmMonitoringService(edm::ParameterSet const& config, edm::ActivityRegistry& registry) {
  // make sure that ROCm is initialised, and that the ROCmService destructor is called after this service's destructor
  edm::Service<ROCmInterface> service;
  if (not service or not service->enabled())
    return;

  numberOfDevices_ = service->numberOfDevices();

  if (config.getUntrackedParameter<bool>("memoryConstruction")) {
    registry.watchPostModuleConstruction(this, &ROCmMonitoringService::postModuleConstruction);
  }
  if (config.getUntrackedParameter<bool>("memoryBeginStream")) {
    registry.watchPostModuleBeginStream(this, &ROCmMonitoringService::postModuleBeginStream);
  }
  if (config.getUntrackedParameter<bool>("memoryPerModule")) {
    registry.watchPostModuleEvent(this, &ROCmMonitoringService::postModuleEvent);
  }
  if (config.getUntrackedParameter<bool>("memoryPerEvent")) {
    registry.watchPostEvent(this, &ROCmMonitoringService::postEvent);
  }
}

void ROCmMonitoringService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<bool>("memoryConstruction", false)
      ->setComment("Print memory information for each device after the construction of each module");
  desc.addUntracked<bool>("memoryBeginStream", true)
      ->setComment("Print memory information for each device after the beginStream() of each module");
  desc.addUntracked<bool>("memoryPerModule", true)
      ->setComment("Print memory information for each device after the event of each module");
  desc.addUntracked<bool>("memoryPerEvent", true)
      ->setComment("Print memory information for each device after each event");

  descriptions.add("ROCmMonitoringService", desc);
  descriptions.setComment(
      "The memory information is the global state of the device. This gets confusing if there are multiple processes "
      "running on the same device. Probably the information retrieval should be re-thought?");
}

// activity handlers
namespace {
  template <typename T>
  void dumpUsedMemory(T& log, int num) {
    int old = 0;
    hipCheck(hipGetDevice(&old));
    constexpr auto mbytes = 1 << 20;
    for (int i = 0; i < num; ++i) {
      size_t freeMemory, totalMemory;
      hipCheck(hipSetDevice(i));
      hipCheck(hipMemGetInfo(&freeMemory, &totalMemory));
      log << "\n"
          << i << ": " << (totalMemory - freeMemory) / mbytes << " MB used / " << totalMemory / mbytes << " MB total";
    }
    hipCheck(hipSetDevice(old));
  }
}  // namespace

void ROCmMonitoringService::postModuleConstruction(edm::ModuleDescription const& desc) {
  auto log = edm::LogPrint("ROCmMonitoringService");
  log << "ROCm device memory after construction of " << desc.moduleLabel() << " (" << desc.moduleName() << ")";
  dumpUsedMemory(log, numberOfDevices_);
}

void ROCmMonitoringService::postModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
  auto log = edm::LogPrint("ROCmMonitoringService");
  log << "ROCm device memory after beginStream() of " << mcc.moduleDescription()->moduleLabel() << " ("
      << mcc.moduleDescription()->moduleName() << ")";
  dumpUsedMemory(log, numberOfDevices_);
}

void ROCmMonitoringService::postModuleEvent(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
  auto log = edm::LogPrint("ROCmMonitoringService");
  log << "ROCm device memory after processing an event by " << mcc.moduleDescription()->moduleLabel() << " ("
      << mcc.moduleDescription()->moduleName() << ")";
  dumpUsedMemory(log, numberOfDevices_);
}

void ROCmMonitoringService::postEvent(edm::StreamContext const& sc) {
  auto log = edm::LogPrint("ROCmMonitoringService");
  log << "ROCm device memory after event";
  dumpUsedMemory(log, numberOfDevices_);
}

DEFINE_FWK_SERVICE(ROCmMonitoringService);
