#include "HeterogeneousCore/AcceleratorService/interface/AcceleratorService.h"

#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include <limits>
#include <algorithm>
#include <thread>
#include <random>
#include <chrono>
#include <cassert>

thread_local unsigned int AcceleratorService::currentModuleId_ = std::numeric_limits<unsigned int>::max();
thread_local std::string AcceleratorService::currentModuleLabel_ = "";

AcceleratorService::AcceleratorService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
  iRegistry.watchPreallocate(           this, &AcceleratorService::preallocate );
  iRegistry.watchPreModuleConstruction (this, &AcceleratorService::preModuleConstruction );
  iRegistry.watchPostModuleConstruction(this, &AcceleratorService::postModuleConstruction );
}

// signals
void AcceleratorService::preallocate(edm::service::SystemBounds const& bounds) {
  numberOfStreams_ = bounds.maxNumberOfStreams();
  LogTrace("AcceleratorService") << "AcceleratorService: number of streams " << numberOfStreams_;
  // called after module construction, so initialize algoExecutionLocation_ here
  algoExecutionLocation_.resize(moduleIds_.size()*numberOfStreams_);
}

void AcceleratorService::preModuleConstruction(edm::ModuleDescription const& desc) {
  currentModuleId_ = desc.id();
  currentModuleLabel_ = desc.moduleLabel();
}
void AcceleratorService::postModuleConstruction(edm::ModuleDescription const& desc) {
  currentModuleId_ = std::numeric_limits<unsigned int>::max();
  currentModuleLabel_ = "";
}


// actual functionality
AcceleratorService::Token AcceleratorService::book() {
  if(currentModuleId_ == std::numeric_limits<unsigned int>::max())
    throw cms::Exception("AcceleratorService") << "Calling AcceleratorService::register() outside of EDModule constructor is forbidden.";

  unsigned int index=0;

  std::lock_guard<std::mutex> guard(moduleMutex_);

  auto found = std::find(moduleIds_.begin(), moduleIds_.end(), currentModuleId_);
  if(found == moduleIds_.end()) {
    index = moduleIds_.size();
    moduleIds_.push_back(currentModuleId_);
  }
  else {
    index = std::distance(moduleIds_.begin(), found);
  }

  LogTrace("AcceleratorService") << "AcceleratorService::book for module " << currentModuleId_ << " " << currentModuleLabel_ << " token id " << index << " moduleIds_.size() " << moduleIds_.size();

  return Token(index);
}

bool AcceleratorService::scheduleGPUMock(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUMockBase& gpuMockAlgo) {
  // Decide randomly whether to run on GPU or CPU to simulate scheduler decisions
  std::random_device r;
  std::mt19937 gen(r());
  auto dist1 = std::uniform_int_distribution<>(0, 10); // simulate the scheduler decision
  if(dist1(gen) == 0) {
    LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " GPUMock is disabled (by chance)";
    return false;
  }

  LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launching task on GPUMock";
  gpuMockAlgo.runGPUMock([waitingTaskHolder = std::move(waitingTaskHolder),
                          token = token,
                          streamID = streamID,
                          &location = algoExecutionLocation_[tokenStreamIdsToDataIndex(token.id(), streamID)]
                          ]() mutable {
                           LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " task finished on GPUMock";
                           location = HeterogeneousDeviceId(HeterogeneousDevice::kGPUMock, 0);
                           waitingTaskHolder.doneWaiting(nullptr);
                         });
  LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launched task on GPUMock asynchronously(?)";
  return true;
}

bool AcceleratorService::scheduleGPUCuda(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoGPUCudaBase& gpuCudaAlgo) {
  edm::Service<CUDAService> cudaService;
  if(!cudaService->enabled()) {
    LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " CudaService is disabled";
    return false;
  }

  LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launching task on GPU";
  gpuCudaAlgo.runGPUCuda([waitingTaskHolder = std::move(waitingTaskHolder),
                          token = token,
                          streamID = streamID,
                          &location = algoExecutionLocation_[tokenStreamIdsToDataIndex(token.id(), streamID)]
                          ]() mutable {
                           LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " task finished on GPU";
                           location = HeterogeneousDeviceId(HeterogeneousDevice::kGPUCuda, 0);
                           waitingTaskHolder.doneWaiting(nullptr);
                         });
  LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launched task on GPU asynchronously(?)";
  return true;
}

void AcceleratorService::scheduleCPU(Token token, edm::StreamID streamID, edm::WaitingTaskWithArenaHolder waitingTaskHolder, accelerator::AlgoCPUBase& cpuAlgo) {
  LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " launching task on CPU";
  cpuAlgo.runCPU();
  algoExecutionLocation_[tokenStreamIdsToDataIndex(token.id(), streamID)] = HeterogeneousDeviceId(HeterogeneousDevice::kCPU, 0);
  LogTrace("AcceleratorService") << "  AcceleratorService token " << token.id() << " stream " << streamID << " task finished on CPU";
}


unsigned int AcceleratorService::tokenStreamIdsToDataIndex(unsigned int tokenId, edm::StreamID streamId) const {
  assert(streamId < numberOfStreams_);
  return tokenId*numberOfStreams_ + streamId;
}
