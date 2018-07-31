#include <cuda.h>
#include <cuda/api_wrappers.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/getCudaDrvErrorString.h"

CUDAService::CUDAService(edm::ParameterSet const& config, edm::ActivityRegistry& iRegistry) {
  bool configEnabled = config.getUntrackedParameter<bool>("enabled");
  if (not configEnabled) {
    edm::LogInfo("CUDAService") << "CUDAService disabled by configuration";
    return;
  }

  auto status = cudaGetDeviceCount(&numberOfDevices_);
  if (cudaSuccess != status) {
    edm::LogWarning("CUDAService") << "Failed to initialize the CUDA runtime.\n" << ".\n" << "Disabling the CUDAService.";
    return;
  }
  edm::LogInfo("CUDAService") << "CUDA runtime successfully initialised, found " << numberOfDevices_ << " compute devices";

  auto numberOfStreamsPerDevice = config.getUntrackedParameter<unsigned int>("numberOfStreamsPerDevice");
  if (numberOfStreamsPerDevice > 0) {
    numberOfStreamsTotal_ = numberOfStreamsPerDevice * numberOfDevices_;
    edm::LogSystem("CUDAService") << "Number of edm::Streams per CUDA device has been set to " << numberOfStreamsPerDevice << ". With " << numberOfDevices_ << " CUDA devices, this means total of " << numberOfStreamsTotal_ << " edm::Streams for all CUDA devices."; // TODO: eventually silence to LogDebug
  }

  computeCapabilities_.reserve(numberOfDevices_);
  for (int i = 0; i < numberOfDevices_; ++i) {
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, i)); 
    edm::LogInfo("CUDAService") << "Device " << i << " with compute capability " << properties.major << "." << properties.minor;
    computeCapabilities_.emplace_back(properties.major, properties.minor);
  }

  edm::LogInfo("CUDAService") << "CUDAService fully initialized";
  enabled_ = true;
}

CUDAService::~CUDAService() {
  if (enabled_) {
    for (int i = 0; i < numberOfDevices_; ++i) {
      cudaCheck(cudaSetDevice(i));
      cudaCheck(cudaDeviceSynchronize());
      // Explicitly destroys and cleans up all resources associated with the current device in the
      // current process. Any subsequent API call to this device will reinitialize the device.
      // Useful to check for memory leaks with `cuda-memcheck --tool memcheck --leak-check full`.
      cudaDeviceReset();
    }
  }
}

void CUDAService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("enabled", true);
  desc.addUntracked<unsigned int>("numberOfStreamsPerDevice", 0)->setComment("Upper limit of the number of edm::Streams that will run on a single CUDA GPU device. The remaining edm::Streams will be run only on other devices (for time being this means CPU in practice). The value '0' means 'unlimited', a value >= 1 imposes the limit.");

  descriptions.add("CUDAService", desc);
}

int CUDAService::deviceWithMostFreeMemory() const {
  size_t freeMem = 0;
  int devId = -1;
  for(int i = 0; i < numberOfDevices_; ++i) {
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
