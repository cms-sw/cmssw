#include <iostream>

#include <cuda.h>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

namespace edm {
  class StreamContext;
}

class CUDAMonitoringService {
public:
  CUDAMonitoringService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry);
  ~CUDAMonitoringService() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void postModuleConstruction(edm::ModuleDescription const& desc);
  void postModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const& mcc);
  void postEvent(edm::StreamContext const& sc);

private:
  int numberOfDevices_ = 0;
};

CUDAMonitoringService::CUDAMonitoringService(edm::ParameterSet const& config, edm::ActivityRegistry& registry) {
  // make sure that CUDA is initialised, and that the CUDAService destructor is called after this service's destructor
  edm::Service<CUDAService> cudaService;
  if (!cudaService->enabled())
    return;
  numberOfDevices_ = cudaService->numberOfDevices();

  if (config.getUntrackedParameter<bool>("memoryConstruction")) {
    registry.watchPostModuleConstruction(this, &CUDAMonitoringService::postModuleConstruction);
  }
  if (config.getUntrackedParameter<bool>("memoryBeginStream")) {
    registry.watchPostModuleBeginStream(this, &CUDAMonitoringService::postModuleBeginStream);
  }
  if (config.getUntrackedParameter<bool>("memoryPerEvent")) {
    registry.watchPostEvent(this, &CUDAMonitoringService::postEvent);
  }
}

void CUDAMonitoringService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<bool>("memoryConstruction", false)
      ->setComment("Print memory information for each device after the construction of each module");
  desc.addUntracked<bool>("memoryBeginStream", true)
      ->setComment("Print memory information for each device after the beginStream() of each module");
  desc.addUntracked<bool>("memoryPerEvent", true)
      ->setComment("Print memory information for each device after each event");

  descriptions.add("CUDAMonitoringService", desc);
  descriptions.setComment(
      "The memory information is the global state of the device. This gets confusing if there are multiple processes "
      "running on the same device. Probably the information retrieval should be re-thought?");
}

// activity handlers
namespace {
  template <typename T>
  void dumpUsedMemory(T& log, int num) {
    int old = 0;
    cudaCheck(cudaGetDevice(&old));
    for (int i = 0; i < num; ++i) {
      size_t freeMemory, totalMemory;
      cudaCheck(cudaSetDevice(i));
      cudaCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
      log << "\n"
          << i << ": " << (totalMemory - freeMemory) / (1 << 20) << " MB used / " << totalMemory / (1 << 20)
          << " MB total";
    }
    cudaCheck(cudaSetDevice(old));
  }
}  // namespace

void CUDAMonitoringService::postModuleConstruction(edm::ModuleDescription const& desc) {
  auto log = edm::LogPrint("CUDAMonitoringService");
  log << "CUDA device memory after construction of " << desc.moduleLabel() << " (" << desc.moduleName() << ")";
  dumpUsedMemory(log, numberOfDevices_);
}

void CUDAMonitoringService::postModuleBeginStream(edm::StreamContext const&, edm::ModuleCallingContext const& mcc) {
  auto log = edm::LogPrint("CUDAMonitoringService");
  log << "CUDA device memory after beginStream() of " << mcc.moduleDescription()->moduleLabel() << " ("
      << mcc.moduleDescription()->moduleName() << ")";
  dumpUsedMemory(log, numberOfDevices_);
}

void CUDAMonitoringService::postEvent(edm::StreamContext const& sc) {
  auto log = edm::LogPrint("CUDAMonitoringService");
  log << "CUDA device memory after event";
  dumpUsedMemory(log, numberOfDevices_);
}

DEFINE_FWK_SERVICE(CUDAMonitoringService);
