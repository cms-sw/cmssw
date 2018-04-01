#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <cuda.h>

#include <dlfcn.h>

CUDAService::CUDAService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
  bool configEnabled = iConfig.getUntrackedParameter<bool>("enabled");
  if(!configEnabled) {
    edm::LogInfo("CUDAService") << "CUDAService disabled by configuration";
  }

  // First check if we can load the cuda runtime library
  void *cudaLib = dlopen("libcuda.so", RTLD_NOW);
  if(cudaLib == nullptr) {
    edm::LogWarning("CUDAService") << "Failed to dlopen libcuda.so, disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "Found libcuda.so";

  // Find functions
  auto cuInit = reinterpret_cast<CUresult (*)(unsigned int Flags)>(dlsym(cudaLib, "cuInit"));
  if(cuInit == nullptr) {
    edm::LogWarning("CUDAService") << "Failed to find cuInit from libcuda.so, disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "Found cuInit from libcuda";

  auto cuDeviceGetCount = reinterpret_cast<CUresult (*)(int *count)>(dlsym(cudaLib, "cuDeviceGetCount"));
  if(cuDeviceGetCount == nullptr) {
    edm::LogWarning("CUDAService") << "Failed to find cuDeviceGetCount from libcuda.so, disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "Found cuDeviceGetCount from libcuda";

  auto cuDeviceComputeCapability = reinterpret_cast<CUresult (*)(int *major, int *minor, CUdevice dev)>(dlsym(cudaLib, "cuDeviceComputeCapability"));
  if(cuDeviceComputeCapability == nullptr) {
    edm::LogWarning("CUDAService") << "Failed to find cuDeviceComputeCapability from libcuda.so, disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "Found cuDeviceComputeCapability";

  // Then call functions
  auto ret = cuInit(0);
  if(CUDA_SUCCESS != ret) {
    edm::LogWarning("CUDAService") << "cuInit failed, return value " << ret << ", disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "cuInit succeeded";

  ret = cuDeviceGetCount(&numberOfDevices_);
  if(CUDA_SUCCESS != ret) {
    edm::LogWarning("CUDAService") << "cuDeviceGetCount failed, return value " << ret << ", disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "cuDeviceGetCount succeeded, found " << numberOfDevices_ << " devices";
  if(numberOfDevices_ < 1) {
    edm::LogWarning("CUDAService") << "Number of devices < 1, disabling CUDAService";
    return;
  }

  computeCapabilities_.reserve(numberOfDevices_);
  for(int i=0; i<numberOfDevices_; ++i) {
    int major, minor;
    ret = cuDeviceComputeCapability(&major, &minor, i);
    if(CUDA_SUCCESS != ret) {
      edm::LogWarning("CUDAService") << "cuDeviceComputeCapability failed for device " << i << ", return value " << ret << " disabling CUDAService";
      return;
    }
    edm::LogInfo("CUDAService") << "Device " << i << " compute capability major " << major << " minor " << minor;
    computeCapabilities_.emplace_back(major, minor);
  }

  edm::LogInfo("CUDAService") << "CUDAService fully initialized";
  enabled_ = true;
}

void CUDAService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("enabled", true);

  descriptions.add("CUDAService", desc);
}
