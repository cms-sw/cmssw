#include <iomanip>
#include <iostream>
#include <limits>

#include <cuda.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/EventCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/StreamCache.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/currentDevice.h"
#include "HeterogeneousCore/CUDAUtilities/src/getCachingDeviceAllocator.h"
#include "HeterogeneousCore/CUDAUtilities/src/getCachingHostAllocator.h"

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
    edm::LogWarning("CUDAService") << "CUDA device " << device << ": failed to set limit \"" << name << "\" to "
                                   << request << ", current value is " << value;
  } else if (value != request) {
    edm::LogWarning("CUDAService") << "CUDA device " << device << ": limit \"" << name << "\" set to " << value
                                   << " instead of requested " << request;
  }
}

constexpr unsigned int getCudaCoresPerSM(unsigned int major, unsigned int minor) {
  switch (major * 10 + minor) {
    // Fermi architecture
    case 20:  // SM 2.0: GF100 class
      return 32;
    case 21:  // SM 2.1: GF10x class
      return 48;

    // Kepler architecture
    case 30:  // SM 3.0: GK10x class
    case 32:  // SM 3.2: GK10x class
    case 35:  // SM 3.5: GK11x class
    case 37:  // SM 3.7: GK21x class
      return 192;

    // Maxwell architecture
    case 50:  // SM 5.0: GM10x class
    case 52:  // SM 5.2: GM20x class
    case 53:  // SM 5.3: GM20x class
      return 128;

    // Pascal architecture
    case 60:  // SM 6.0: GP100 class
      return 64;
    case 61:  // SM 6.1: GP10x class
    case 62:  // SM 6.2: GP10x class
      return 128;

    // Volta architecture
    case 70:  // SM 7.0: GV100 class
    case 72:  // SM 7.2: GV11b class
      return 64;

    // Turing architecture
    case 75:  // SM 7.5: TU10x class
      return 64;

    // unknown architecture, return a default value
    default:
      return 64;
  }
}

namespace {
  template <template <typename> typename UniquePtr, typename Allocate>
  void preallocate(Allocate allocate, const std::vector<unsigned int>& bufferSizes) {
    if (bufferSizes.empty())
      return;

    auto streamPtr = cms::cuda::getStreamCache().get();

    std::vector<UniquePtr<char[]> > buffers;
    buffers.reserve(bufferSizes.size());
    for (auto size : bufferSizes) {
      buffers.push_back(allocate(size, streamPtr.get()));
    }
  }

  void devicePreallocate(int numberOfDevices, const std::vector<unsigned int>& bufferSizes) {
    int device;
    cudaCheck(cudaGetDevice(&device));
    for (int i = 0; i < numberOfDevices; ++i) {
      cudaCheck(cudaSetDevice(i));
      preallocate<cms::cuda::device::unique_ptr>(
          [&](size_t size, cudaStream_t stream) { return cms::cuda::make_device_unique<char[]>(size, stream); },
          bufferSizes);
    }
    cudaCheck(cudaSetDevice(device));
  }

  void hostPreallocate(const std::vector<unsigned int>& bufferSizes) {
    preallocate<cms::cuda::host::unique_ptr>(
        [&](size_t size, cudaStream_t stream) { return cms::cuda::make_host_unique<char[]>(size, stream); },
        bufferSizes);
  }
}  // namespace

/// Constructor
CUDAService::CUDAService(edm::ParameterSet const& config) {
  bool configEnabled = config.getUntrackedParameter<bool>("enabled");
  if (not configEnabled) {
    edm::LogInfo("CUDAService") << "CUDAService disabled by configuration";
    return;
  }

  auto status = cudaGetDeviceCount(&numberOfDevices_);
  if (cudaSuccess != status) {
    edm::LogWarning("CUDAService") << "Failed to initialize the CUDA runtime.\n"
                                   << "Disabling the CUDAService.";
    return;
  }
  edm::LogInfo log("CUDAService");
  computeCapabilities_.reserve(numberOfDevices_);
  log << "CUDA runtime successfully initialised, found " << numberOfDevices_ << " compute devices.\n\n";

  auto const& limits = config.getUntrackedParameter<edm::ParameterSet>("limits");
  auto printfFifoSize = limits.getUntrackedParameter<int>("cudaLimitPrintfFifoSize");
  auto stackSize = limits.getUntrackedParameter<int>("cudaLimitStackSize");
  auto mallocHeapSize = limits.getUntrackedParameter<int>("cudaLimitMallocHeapSize");
  auto devRuntimeSyncDepth = limits.getUntrackedParameter<int>("cudaLimitDevRuntimeSyncDepth");
  auto devRuntimePendingLaunchCount = limits.getUntrackedParameter<int>("cudaLimitDevRuntimePendingLaunchCount");

  for (int i = 0; i < numberOfDevices_; ++i) {
    // read information about the compute device.
    // see the documentation of cudaGetDeviceProperties() for more information.
    cudaDeviceProp properties;
    cudaCheck(cudaGetDeviceProperties(&properties, i));
    log << "CUDA device " << i << ": " << properties.name << '\n';

    // compute capabilities
    log << "  compute capability:          " << properties.major << "." << properties.minor << " (sm_"
        << properties.major << properties.minor << ")\n";
    computeCapabilities_.emplace_back(properties.major, properties.minor);
    log << "  streaming multiprocessors: " << std::setw(13) << properties.multiProcessorCount << '\n';
    log << "  CUDA cores: " << std::setw(28)
        << properties.multiProcessorCount * getCudaCoresPerSM(properties.major, properties.minor) << '\n';
    log << "  single to double performance: " << std::setw(8) << properties.singleToDoublePrecisionPerfRatio << ":1\n";

    // compute mode
    static constexpr const char* computeModeDescription[] = {
        "default (shared)",            // cudaComputeModeDefault
        "exclusive (single thread)",   // cudaComputeModeExclusive
        "prohibited",                  // cudaComputeModeProhibited
        "exclusive (single process)",  // cudaComputeModeExclusiveProcess
        "unknown"};
    log << "  compute mode:" << std::right << std::setw(27)
        << computeModeDescription[std::min(properties.computeMode,
                                           static_cast<int>(std::size(computeModeDescription)) - 1)]
        << '\n';

    // TODO if a device is in exclusive use, skip it and remove it from the list, instead of failing with abort()
    cudaCheck(cudaSetDevice(i));
    cudaCheck(cudaSetDeviceFlags(cudaDeviceScheduleAuto | cudaDeviceMapHost));

    // read the free and total amount of memory available for allocation by the device, in bytes.
    // see the documentation of cudaMemGetInfo() for more information.
    size_t freeMemory, totalMemory;
    cudaCheck(cudaMemGetInfo(&freeMemory, &totalMemory));
    log << "  memory: " << std::setw(6) << freeMemory / (1 << 20) << " MB free / " << std::setw(6)
        << totalMemory / (1 << 20) << " MB total\n";
    log << "  constant memory:               " << std::setw(6) << properties.totalConstMem / (1 << 10) << " kB\n";
    log << "  L2 cache size:                 " << std::setw(6) << properties.l2CacheSize / (1 << 10) << " kB\n";

    // L1 cache behaviour
    static constexpr const char* l1CacheModeDescription[] = {
        "unknown", "local memory", "global memory", "local and global memory"};
    int l1CacheMode = properties.localL1CacheSupported + 2 * properties.globalL1CacheSupported;
    log << "  L1 cache mode:" << std::setw(26) << std::right << l1CacheModeDescription[l1CacheMode] << '\n';
    log << '\n';

    log << "Other capabilities\n";
    log << "  " << (properties.canMapHostMemory ? "can" : "cannot")
        << " map host memory into the CUDA address space for use with cudaHostAlloc()/cudaHostGetDevicePointer()\n";
    log << "  " << (properties.pageableMemoryAccess ? "supports" : "does not support")
        << " coherently accessing pageable memory without calling cudaHostRegister() on it\n";
    log << "  " << (properties.pageableMemoryAccessUsesHostPageTables ? "can" : "cannot")
        << " access pageable memory via the host's page tables\n";
    log << "  " << (properties.canUseHostPointerForRegisteredMem ? "can" : "cannot")
        << " access host registered memory at the same virtual address as the host\n";
    log << "  " << (properties.unifiedAddressing ? "shares" : "does not share")
        << " a unified address space with the host\n";
    log << "  " << (properties.managedMemory ? "supports" : "does not support")
        << " allocating managed memory on this system\n";
    log << "  " << (properties.concurrentManagedAccess ? "can" : "cannot")
        << " coherently access managed memory concurrently with the host\n";
    log << "  "
        << "the host " << (properties.directManagedMemAccessFromHost ? "can" : "cannot")
        << " directly access managed memory on the device without migration\n";
    log << "  " << (properties.cooperativeLaunch ? "supports" : "does not support")
        << " launching cooperative kernels via cudaLaunchCooperativeKernel()\n";
    log << "  " << (properties.cooperativeMultiDeviceLaunch ? "supports" : "does not support")
        << " launching cooperative kernels via cudaLaunchCooperativeKernelMultiDevice()\n";
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
        setCudaLimit(cudaLimitDevRuntimePendingLaunchCount,
                     "cudaLimitDevRuntimePendingLaunchCount",
                     devRuntimePendingLaunchCount);
      }
    }

    size_t value;
    log << "CUDA limits\n";
    cudaCheck(cudaDeviceGetLimit(&value, cudaLimitPrintfFifoSize));
    log << "  printf buffer size:        " << std::setw(10) << value / (1 << 20) << " MB\n";
    cudaCheck(cudaDeviceGetLimit(&value, cudaLimitStackSize));
    log << "  stack size:                " << std::setw(10) << value / (1 << 10) << " kB\n";
    cudaCheck(cudaDeviceGetLimit(&value, cudaLimitMallocHeapSize));
    log << "  malloc heap size:          " << std::setw(10) << value / (1 << 20) << " MB\n";
    if ((properties.major > 3) or (properties.major == 3 and properties.minor >= 5)) {
      cudaCheck(cudaDeviceGetLimit(&value, cudaLimitDevRuntimeSyncDepth));
      log << "  runtime sync depth:           " << std::setw(10) << value << '\n';
      cudaCheck(cudaDeviceGetLimit(&value, cudaLimitDevRuntimePendingLaunchCount));
      log << "  runtime pending launch count: " << std::setw(10) << value << '\n';
    }
    log << '\n';
  }
  log << "\n";

  // Make sure the caching allocators and stream/event caches are constructed before declaring successful construction
  if constexpr (cms::cuda::allocator::useCaching) {
    cms::cuda::allocator::getCachingDeviceAllocator();
    cms::cuda::allocator::getCachingHostAllocator();
  }
  cms::cuda::getEventCache().clear();
  cms::cuda::getStreamCache().clear();

  log << "CUDAService fully initialized";
  enabled_ = true;

  // Preallocate buffers if asked to
  auto const& allocator = config.getUntrackedParameter<edm::ParameterSet>("allocator");
  devicePreallocate(numberOfDevices_, allocator.getUntrackedParameter<std::vector<unsigned int> >("devicePreallocate"));
  hostPreallocate(allocator.getUntrackedParameter<std::vector<unsigned int> >("hostPreallocate"));
}

CUDAService::~CUDAService() {
  if (enabled_) {
    // Explicitly destruct the allocator before the device resets below
    if constexpr (cms::cuda::allocator::useCaching) {
      cms::cuda::allocator::getCachingDeviceAllocator().FreeAllCached();
      cms::cuda::allocator::getCachingHostAllocator().FreeAllCached();
    }
    cms::cuda::getEventCache().clear();
    cms::cuda::getStreamCache().clear();

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

void CUDAService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("enabled", true);

  edm::ParameterSetDescription limits;
  limits.addUntracked<int>("cudaLimitPrintfFifoSize", -1)
      ->setComment("Size in bytes of the shared FIFO used by the printf() device system call.");
  limits.addUntracked<int>("cudaLimitStackSize", -1)->setComment("Stack size in bytes of each GPU thread.");
  limits.addUntracked<int>("cudaLimitMallocHeapSize", -1)
      ->setComment("Size in bytes of the heap used by the malloc() and free() device system calls.");
  limits.addUntracked<int>("cudaLimitDevRuntimeSyncDepth", -1)
      ->setComment("Maximum nesting depth of a grid at which a thread can safely call cudaDeviceSynchronize().");
  limits.addUntracked<int>("cudaLimitDevRuntimePendingLaunchCount", -1)
      ->setComment("Maximum number of outstanding device runtime launches that can be made from the current device.");
  desc.addUntracked<edm::ParameterSetDescription>("limits", limits)
      ->setComment(
          "See the documentation of cudaDeviceSetLimit for more information.\nSetting any of these options to -1 keeps "
          "the default value.");

  edm::ParameterSetDescription allocator;
  allocator.addUntracked<std::vector<unsigned int> >("devicePreallocate", std::vector<unsigned int>{})
      ->setComment("Preallocates buffers of given bytes on all devices");
  allocator.addUntracked<std::vector<unsigned int> >("hostPreallocate", std::vector<unsigned int>{})
      ->setComment("Preallocates buffers of given bytes on the host");
  desc.addUntracked<edm::ParameterSetDescription>("allocator", allocator);

  descriptions.add("CUDAService", desc);
}

int CUDAService::deviceWithMostFreeMemory() const {
  // save the current device
  int currentDevice;
  cudaCheck(cudaGetDevice(&currentDevice));

  size_t maxFreeMemory = 0;
  int device = -1;
  for (int i = 0; i < numberOfDevices_; ++i) {
    size_t freeMemory, totalMemory;
    cudaSetDevice(i);
    cudaMemGetInfo(&freeMemory, &totalMemory);
    edm::LogPrint("CUDAService") << "CUDA device " << i << ": " << freeMemory / (1 << 20) << " MB free / "
                                 << totalMemory / (1 << 20) << " MB total memory";
    if (freeMemory > maxFreeMemory) {
      maxFreeMemory = freeMemory;
      device = i;
    }
  }
  // restore the current device
  cudaCheck(cudaSetDevice(currentDevice));
  return device;
}
