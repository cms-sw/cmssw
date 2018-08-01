#include <iomanip>
#include <iostream>

#include <cuda.h>
#include <cuda/api_wrappers.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"

void setCudaLimit(cudaLimit limit, const char* name, size_t request) {
  // read the current device
  int device;
  cudaCheck(cudaGetDevice(&device));
  // try to set the requested limit
  auto result = cudaDeviceSetLimit(limit, request);
  if (cudaErrorUnsupportedLimit == result) {
    edm::LogWarning("CUDAService") << "CUDA device " << device << ": unsupported limit \"" << name << "\"";
    return;
  }
  // read back the limit value
  size_t value;
  cudaCheck(cudaDeviceGetLimit(&value, limit));
  if (cudaSuccess != result) {
    edm::LogWarning("CUDAService") << "CUDA device " << device << ": failed to set limit \"" << name << "\" to " << request << ", current value is " << value ;
  } else if (value != request) {
    edm::LogWarning("CUDAService") << "CUDA device " << device << ": limit \"" << name << "\" set to " << value << " instead of requested " << request;
  }
}

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
  edm::LogInfo log("CUDAService");
  computeCapabilities_.reserve(numberOfDevices_);
  log << "CUDA runtime successfully initialised, found " << numberOfDevices_ << " compute devices.\n\n";

  auto numberOfStreamsPerDevice = config.getUntrackedParameter<unsigned int>("numberOfStreamsPerDevice");
  if (numberOfStreamsPerDevice > 0) {
    numberOfStreamsTotal_ = numberOfStreamsPerDevice * numberOfDevices_;
    log << "Number of edm::Streams per CUDA device has been set to " << numberOfStreamsPerDevice << ", for a total of " << numberOfStreamsTotal_ << " edm::Streams across all CUDA device(s).\n\n";
  }

  auto const& limits = config.getUntrackedParameter<edm::ParameterSet>("limits");
  auto printfFifoSize               = limits.getUntrackedParameter<int>("cudaLimitPrintfFifoSize");
  auto stackSize                    = limits.getUntrackedParameter<int>("cudaLimitStackSize");
  auto mallocHeapSize               = limits.getUntrackedParameter<int>("cudaLimitMallocHeapSize");
  auto devRuntimeSyncDepth          = limits.getUntrackedParameter<int>("cudaLimitDevRuntimeSyncDepth");
  auto devRuntimePendingLaunchCount = limits.getUntrackedParameter<int>("cudaLimitDevRuntimePendingLaunchCount");

  for (int i = 0; i < numberOfDevices_; ++i) {
    // read information about the compute device.
    // see the documentation of cudaGetDeviceProperties() for more information.
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, i));
    log << "CUDA device " << i << ": " << properties.name << '\n';
    log << "  compute capability: " << properties.major << "." << properties.minor << '\n';
    computeCapabilities_.emplace_back(properties.major, properties.minor);

    cudaCheck(cudaSetDevice(i));
    cudaCheck(cudaSetDeviceFlags(cudaDeviceScheduleAuto | cudaDeviceMapHost));

    // read the free and total amount of memory available for allocation by the device, in bytes.
    // see the documentation of cudaMemGetInfo() for more information.
    size_t freeMemory, totalMemory;
    cudaCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
    log << "  memory: " << std::setw(6) << freeMemory / (1 << 20) << " MB free / " << std::setw(6) << totalMemory / (1 << 20) << " MB total\n";
    log << '\n';

    // set and read the CUDA device flags.
    // see the documentation of cudaSetDeviceFlags and cudaGetDeviceFlags for  more information.
    log << "CUDA flags\n";
    unsigned int flags;
    cudaCheck(cudaGetDeviceFlags(&flags));
    switch (flags & cudaDeviceScheduleMask) {
      case cudaDeviceScheduleAuto:
        log << "  thread policy:                   default\n";
        break;
      case cudaDeviceScheduleSpin:
        log << "  thread policy:                      spin\n";
        break;
      case cudaDeviceScheduleYield:
        log << "  thread policy:                     yield\n";
        break;
      case cudaDeviceScheduleBlockingSync:
        log << "  thread policy:             blocking sync\n";
        break;
      default:
        log << "  thread policy:                 undefined\n";
    }
    if (flags & cudaDeviceMapHost) {
      log << "  pinned host memory allocations:  enabled\n";
    } else {
      log << "  pinned host memory allocations: disabled\n";
    }
    if (flags & cudaDeviceLmemResizeToMax) {
      log << "  kernel host memory reuse:        enabled\n";
    } else {
      log << "  kernel host memory reuse:       disabled\n";
    }
    log << '\n';

    // set and read the CUDA resource limits.
    // see the documentation of cudaDeviceSetLimit() for more information.

    // cudaLimitPrintfFifoSize controls the size in bytes of the shared FIFO used by the
    // printf() device system call.
    if (printfFifoSize >= 0) {
      setCudaLimit(cudaLimitPrintfFifoSize, "cudaLimitPrintfFifoSize", printfFifoSize);
    }
    // cudaLimitStackSize controls the stack size in bytes of each GPU thread.
    if (stackSize >= 0) {
      setCudaLimit(cudaLimitStackSize, "cudaLimitStackSize", stackSize);
    }
    // cudaLimitMallocHeapSize controls the size in bytes of the heap used by the malloc()
    // and free() device system calls.
    if (mallocHeapSize >= 0) {
      setCudaLimit(cudaLimitMallocHeapSize, "cudaLimitMallocHeapSize", mallocHeapSize);
    }
    if ((properties.major > 3) or (properties.major == 3 and properties.minor >= 5)) {
      // cudaLimitDevRuntimeSyncDepth controls the maximum nesting depth of a grid at which
      // a thread can safely call cudaDeviceSynchronize().
      if (devRuntimeSyncDepth >= 0) {
        setCudaLimit(cudaLimitDevRuntimeSyncDepth, "cudaLimitDevRuntimeSyncDepth", devRuntimeSyncDepth);
      }
      // cudaLimitDevRuntimePendingLaunchCount controls the maximum number of outstanding
      // device runtime launches that can be made from the current device.
      if (devRuntimePendingLaunchCount >= 0) {
        setCudaLimit(cudaLimitDevRuntimePendingLaunchCount, "cudaLimitDevRuntimePendingLaunchCount", devRuntimePendingLaunchCount);
      }
    }

    size_t value;
    log << "CUDA limits\n";
    cudaCheck(cudaDeviceGetLimit(&value, cudaLimitPrintfFifoSize));
    log << "  printf buffer size:           " << std::setw(10) << value << '\n';
    cudaCheck(cudaDeviceGetLimit(&value, cudaLimitStackSize));
    log << "  stack size:                   " << std::setw(10) << value << '\n';
    cudaCheck(cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize));
    log << "  malloc heap size:             " << std::setw(10) << value << '\n';
    if ((properties.major > 3) or (properties.major == 3 and properties.minor >= 5)) {
      cudaCheck(cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth));
      log << "  runtime sync depth:           " << std::setw(10) << value << '\n';
      cudaCheck(cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount));
      log << "  runtime pending launch count: " << std::setw(10) << value << '\n';
    }
    log << '\n';
  }

  log << "CUDAService fully initialized";
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
  desc.addUntracked<unsigned int>("numberOfStreamsPerDevice", 0)->setComment("Upper limit of the number of edm::Streams that will run on a single CUDA GPU device. The remaining edm::Streams will be run only on other devices (for time being this means CPU in practice).\nThe value '0' means 'unlimited', a value >= 1 imposes the limit.");

  edm::ParameterSetDescription limits;
  limits.addUntracked<int>("cudaLimitPrintfFifoSize", -1)->setComment("Size in bytes of the shared FIFO used by the printf() device system call.");
  limits.addUntracked<int>("cudaLimitStackSize", -1)->setComment("Stack size in bytes of each GPU thread.");
  limits.addUntracked<int>("cudaLimitMallocHeapSize", -1)->setComment("Size in bytes of the heap used by the malloc() and free() device system calls.");
  limits.addUntracked<int>("cudaLimitDevRuntimeSyncDepth", -1)->setComment("Maximum nesting depth of a grid at which a thread can safely call cudaDeviceSynchronize().");
  limits.addUntracked<int>("cudaLimitDevRuntimePendingLaunchCount", -1)->setComment("Maximum number of outstanding device runtime launches that can be made from the current device.");
  desc.addUntracked<edm::ParameterSetDescription>("limits", limits)->setComment("See the documentation of cudaDeviceSetLimit for more information.\nSetting any of these options to -1 keeps the default value.");

  descriptions.add("CUDAService", desc);
}

int CUDAService::deviceWithMostFreeMemory() const {
  // save the current device
  int currentDevice;
  cudaCheck(cudaGetDevice(&currentDevice));

  size_t maxFreeMemory = 0;
  int device = -1;
  for(int i = 0; i < numberOfDevices_; ++i) {
    /*
    // TODO: understand why the api-wrappers version gives same value for all devices
    auto device = cuda::device::get(i);
    auto freeMemory = device.memory.amount_free();
    */
    size_t freeMemory, totalMemory;
    cudaSetDevice(i);
    cudaMemGetInfo(&freeMemory, &totalMemory);
    edm::LogPrint("CUDAService") << "CUDA device " << i << ": " << freeMemory / (1 << 20) << " MB free / " << totalMemory / (1 << 20) << " MB total memory";
    if (freeMemory > maxFreeMemory) {
      maxFreeMemory = freeMemory;
      device = i;
    }
  }
  // restore the current device
  cudaCheck(cudaSetDevice(currentDevice));
  return device;
}

void CUDAService::setCurrentDevice(int device) const {
  cuda::device::current::set(device);
}

int CUDAService::getCurrentDevice() const {
  return cuda::device::current::get().id();
}
