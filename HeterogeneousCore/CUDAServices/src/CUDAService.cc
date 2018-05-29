#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include <cuda.h>
#include <cuda/api_wrappers.h>
#include "HeterogeneousCore/CUDAUtilities/interface/getCudaDrvErrorString.h"

#include <dlfcn.h>

CUDAService::CUDAService(edm::ParameterSet const& iConfig, edm::ActivityRegistry& iRegistry) {
  bool configEnabled = iConfig.getUntrackedParameter<bool>("enabled");
  if(!configEnabled) {
    edm::LogInfo("CUDAService") << "CUDAService disabled by configuration";
    return;
  }

  auto ret = cuInit(0);
  if(CUDA_SUCCESS != ret) {
    edm::LogWarning("CUDAService") << "Failed to initialize the CUDA driver API by calling cuInit, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
    return;
  }
  edm::LogInfo("CUDAService") << "cuInit succeeded";

  ret = cuDeviceGetCount(&numberOfDevices_);
  if(CUDA_SUCCESS != ret) {
    edm::LogWarning("CUDAService") << "Failed to call cuDeviceGetCount from CUDA driver API, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
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
      edm::LogWarning("CUDAService") << "Failed to call cuDeviceComputeCapability for device " << i << " from CUDA driver API, return value " << ret << " ("<< getCudaDrvErrorString(ret) << "), disabling CUDAService";
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

int CUDAService::deviceWithMostFreeMemory() const {
  size_t freeMem = 0;
  int devId = -1;
  for(int i=0; i<numberOfDevices_; ++i) {
    // TODO: understand why the api-wrappers version gives same value for all devices
    /*
    auto device = cuda::device::get(i);
    auto mem = device.memory.amount_free();
    */
    size_t free, tot;
    cudaSetDevice(i);
    cudaMemGetInfo(&free, &tot);
    auto mem = free;
    edm::LogPrint("CUDAService") << "Device " << i << " free memory " << mem;
    if(mem > freeMem) {
      freeMem = mem;
      devId = i;
    }
  }
  return devId;
}

void CUDAService::setCurrentDevice(int device) const {
  cuda::device::current::set(device);
}

int CUDAService::getCurrentDevice() const {
  return cuda::device::current::get().id();
}
