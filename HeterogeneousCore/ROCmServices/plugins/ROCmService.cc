#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
/*
#include <nvml.h>
*/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ResourceInformation.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"
#include "HeterogeneousCore/ROCmUtilities/interface/hipCheck.h"
/*
#include "HeterogeneousCore/ROCmUtilities/interface/nvmlCheck.h"
*/

class ROCmService : public ROCmInterface {
public:
  ROCmService(edm::ParameterSet const& config);
  ~ROCmService() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  bool enabled() const final { return enabled_; }

  int numberOfDevices() const final { return numberOfDevices_; }

  // Return the (major, minor) compute capability of the given device.
  std::pair<int, int> computeCapability(int device) const final {
    int size = computeCapabilities_.size();
    if (device < 0 or device >= size) {
      throw std::out_of_range("Invalid device index" + std::to_string(device) + ": the valid range is from 0 to " +
                              std::to_string(size - 1));
    }
    return computeCapabilities_[device];
  }

private:
  int numberOfDevices_ = 0;
  std::vector<std::pair<int, int>> computeCapabilities_;
  bool enabled_ = false;
  bool verbose_ = false;
};

void setHipLimit(hipLimit_t limit, const char* name, size_t request) {
  // read the current device
  int device;
  hipCheck(hipGetDevice(&device));
  // try to set the requested limit
  auto result = hipDeviceSetLimit(limit, request);
  if (hipErrorUnsupportedLimit == result) {
    edm::LogWarning("ROCmService") << "ROCm device " << device << ": unsupported limit \"" << name << "\"";
    return;
  }
  // read back the limit value
  size_t value;
  result = hipDeviceGetLimit(&value, limit);
  if (hipSuccess != result) {
    edm::LogWarning("ROCmService") << "ROCm device " << device << ": failed to set limit \"" << name << "\" to "
                                   << request << ", current value is " << value;
  } else if (value != request) {
    edm::LogWarning("ROCmService") << "ROCm device " << device << ": limit \"" << name << "\" set to " << value
                                   << " instead of requested " << request;
  }
}

std::string decodeVersion(int version) {
  return std::to_string(version / 1000) + '.' + std::to_string(version % 1000 / 10);
}

/// Constructor
ROCmService::ROCmService(edm::ParameterSet const& config) : verbose_(config.getUntrackedParameter<bool>("verbose")) {
  if (not config.getUntrackedParameter<bool>("enabled")) {
    edm::LogInfo("ROCmService") << "ROCmService disabled by configuration";
    return;
  }

  auto status = hipGetDeviceCount(&numberOfDevices_);
  if (hipSuccess != status) {
    edm::LogWarning("ROCmService") << "Failed to initialize the ROCm runtime.\n"
                                   << "Disabling the ROCmService.";
    return;
  }
  computeCapabilities_.reserve(numberOfDevices_);

  /*
  // AMD system driver version, e.g. 470.57.02
  char systemDriverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
  nvmlCheck(nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS | NVML_INIT_FLAG_NO_ATTACH));
  nvmlCheck(nvmlSystemGetDriverVersion(systemDriverVersion, sizeof(systemDriverVersion)));
  nvmlCheck(nvmlShutdown());
  */

  // ROCm driver version, e.g. 11.4
  // the full version, like 11.4.1 or 11.4.100, is not reported
  int driverVersion = 0;
  hipCheck(hipDriverGetVersion(&driverVersion));

  // ROCm runtime version, e.g. 11.4
  // the full version, like 11.4.1 or 11.4.108, is not reported
  int runtimeVersion = 0;
  hipCheck(hipRuntimeGetVersion(&runtimeVersion));

  edm::LogInfo log("ROCmService");
  if (verbose_) {
    /*
    log << "AMD driver:       " << systemDriverVersion << '\n';
    */
    log << "ROCm driver API:  " << decodeVersion(driverVersion) << /*" (compiled with " << decodeVersion(ROCm_VERSION)
        << ")" */
        "\n";
    log << "ROCm runtime API: " << decodeVersion(runtimeVersion)
        << /*" (compiled with " << decodeVersion(ROCmRT_VERSION)
        << ")" */
        "\n";
    log << "ROCm runtime successfully initialised, found " << numberOfDevices_ << " compute devices.\n";
  } else {
    log << "ROCm runtime version " << decodeVersion(runtimeVersion) << ", driver version "
        << decodeVersion(driverVersion)
        /*
        << ", AMD driver version " << systemDriverVersion
        */
        ;
  }

  auto const& limits = config.getUntrackedParameter<edm::ParameterSet>("limits");
  /*
  auto printfFifoSize = limits.getUntrackedParameter<int>("hipLimitPrintfFifoSize");
  */
  auto stackSize = limits.getUntrackedParameter<int>("hipLimitStackSize");
  auto mallocHeapSize = limits.getUntrackedParameter<int>("hipLimitMallocHeapSize");
  /*
  auto devRuntimeSyncDepth = limits.getUntrackedParameter<int>("hipLimitDevRuntimeSyncDepth");
  auto devRuntimePendingLaunchCount = limits.getUntrackedParameter<int>("hipLimitDevRuntimePendingLaunchCount");
  */

  std::set<std::string> models;

  for (int i = 0; i < numberOfDevices_; ++i) {
    // read information about the compute device.
    // see the documentation of hipGetDeviceProperties() for more information.
    hipDeviceProp_t properties;
    hipCheck(hipGetDeviceProperties(&properties, i));
    log << '\n' << "ROCm device " << i << ": " << properties.name;
    if (verbose_) {
      log << '\n';
    }
    models.insert(std::string(properties.name));

    // compute capabilities
    computeCapabilities_.emplace_back(properties.major, properties.minor);
    if (verbose_) {
      log << "  compute capability:          " << properties.major << "." << properties.minor;
    }
    log << " (sm_" << properties.major << properties.minor << ")";
    if (verbose_) {
      log << '\n';
      log << "  streaming multiprocessors: " << std::setw(13) << properties.multiProcessorCount << '\n';
      log << "  ROCm cores: " << std::setw(28) << "not yet implemented" << '\n';
      /*
      log << "  single to double performance: " << std::setw(8) << properties.singleToDoublePrecisionPerfRatio
          << ":1\n";
      */
    }

    // compute mode
    static constexpr const char* computeModeDescription[] = {
        "default (shared)",            // hipComputeModeDefault
        "exclusive (single thread)",   // hipComputeModeExclusive
        "prohibited",                  // hipComputeModeProhibited
        "exclusive (single process)",  // hipComputeModeExclusiveProcess
        "unknown"};
    if (verbose_) {
      log << "  compute mode:" << std::right << std::setw(27)
          << computeModeDescription[std::min(properties.computeMode,
                                             static_cast<int>(std::size(computeModeDescription)) - 1)]
          << '\n';
    }

    // TODO if a device is in exclusive use, skip it and remove it from the list, instead of failing with an exception
    hipCheck(hipSetDevice(i));
    hipCheck(hipSetDeviceFlags(hipDeviceScheduleAuto | hipDeviceMapHost));

    // read the free and total amount of memory available for allocation by the device, in bytes.
    // see the documentation of hipMemGetInfo() for more information.
    if (verbose_) {
      size_t freeMemory, totalMemory;
      hipCheck(hipMemGetInfo(&freeMemory, &totalMemory));
      log << "  memory: " << std::setw(6) << freeMemory / (1 << 20) << " MB free / " << std::setw(6)
          << totalMemory / (1 << 20) << " MB total\n";
      log << "  constant memory:               " << std::setw(6) << properties.totalConstMem / (1 << 10) << " kB\n";
      log << "  L2 cache size:                 " << std::setw(6) << properties.l2CacheSize / (1 << 10) << " kB\n";
    }

    // L1 cache behaviour
    if (verbose_) {
      /*
      static constexpr const char* l1CacheModeDescription[] = {
          "unknown", "local memory", "global memory", "local and global memory"};
      int l1CacheMode = properties.localL1CacheSupported + 2 * properties.globalL1CacheSupported;
      log << "  L1 cache mode:" << std::setw(26) << std::right << l1CacheModeDescription[l1CacheMode] << '\n';
      log << '\n';
      */

      log << "Other capabilities\n";
      log << "  " << (properties.canMapHostMemory ? "can" : "cannot")
          << " map host memory into the ROCm address space for use with hipHostAlloc()/hipHostGetDevicePointer()\n";
      log << "  " << (properties.pageableMemoryAccess ? "supports" : "does not support")
          << " coherently accessing pageable memory without calling hipHostRegister() on it\n";
      log << "  " << (properties.pageableMemoryAccessUsesHostPageTables ? "can" : "cannot")
          << " access pageable memory via the host's page tables\n";
      /*
      log << "  " << (properties.canUseHostPointerForRegisteredMem ? "can" : "cannot")
          << " access host registered memory at the same virtual address as the host\n";
      log << "  " << (properties.unifiedAddressing ? "shares" : "does not share")
          << " a unified address space with the host\n";
      */
      log << "  " << (properties.managedMemory ? "supports" : "does not support")
          << " allocating managed memory on this system\n";
      log << "  " << (properties.concurrentManagedAccess ? "can" : "cannot")
          << " coherently access managed memory concurrently with the host\n";
      log << "  "
          << "the host " << (properties.directManagedMemAccessFromHost ? "can" : "cannot")
          << " directly access managed memory on the device without migration\n";
      log << "  " << (properties.cooperativeLaunch ? "supports" : "does not support")
          << " launching cooperative kernels via hipLaunchCooperativeKernel()\n";
      log << "  " << (properties.cooperativeMultiDeviceLaunch ? "supports" : "does not support")
          << " launching cooperative kernels via hipLaunchCooperativeKernelMultiDevice()\n";
      log << '\n';
    }

    // set and read the ROCm device flags.
    // see the documentation of hipSetDeviceFlags and hipGetDeviceFlags for  more information.
    if (verbose_) {
      log << "ROCm flags\n";
      unsigned int flags;
      hipCheck(hipGetDeviceFlags(&flags));
      switch (flags & hipDeviceScheduleMask) {
        case hipDeviceScheduleAuto:
          log << "  thread policy:                   default\n";
          break;
        case hipDeviceScheduleSpin:
          log << "  thread policy:                      spin\n";
          break;
        case hipDeviceScheduleYield:
          log << "  thread policy:                     yield\n";
          break;
        case hipDeviceScheduleBlockingSync:
          log << "  thread policy:             blocking sync\n";
          break;
        default:
          log << "  thread policy:                 undefined\n";
      }
      if (flags & hipDeviceMapHost) {
        log << "  pinned host memory allocations:  enabled\n";
      } else {
        log << "  pinned host memory allocations: disabled\n";
      }
      if (flags & hipDeviceLmemResizeToMax) {
        log << "  kernel host memory reuse:        enabled\n";
      } else {
        log << "  kernel host memory reuse:       disabled\n";
      }
      log << '\n';
    }

    // set and read the ROCm resource limits.
    // see the documentation of hipDeviceSetLimit() for more information.

    /*
    // hipLimitPrintfFifoSize controls the size in bytes of the shared FIFO used by the
    // printf() device system call.
    if (printfFifoSize >= 0) {
      setHipLimit(hipLimitPrintfFifoSize, "hipLimitPrintfFifoSize", printfFifoSize);
    }
    */
    // hipLimitStackSize controls the stack size in bytes of each GPU thread.
    if (stackSize >= 0) {
      setHipLimit(hipLimitStackSize, "hipLimitStackSize", stackSize);
    }
    // hipLimitMallocHeapSize controls the size in bytes of the heap used by the malloc()
    // and free() device system calls.
    if (mallocHeapSize >= 0) {
      setHipLimit(hipLimitMallocHeapSize, "hipLimitMallocHeapSize", mallocHeapSize);
    }
    /*
    if ((properties.major > 3) or (properties.major == 3 and properties.minor >= 5)) {
      // hipLimitDevRuntimeSyncDepth controls the maximum nesting depth of a grid at which
      // a thread can safely call hipDeviceSynchronize().
      if (devRuntimeSyncDepth >= 0) {
        setHipLimit(hipLimitDevRuntimeSyncDepth, "hipLimitDevRuntimeSyncDepth", devRuntimeSyncDepth);
      }
      // hipLimitDevRuntimePendingLaunchCount controls the maximum number of outstanding
      // device runtime launches that can be made from the current device.
      if (devRuntimePendingLaunchCount >= 0) {
        setHipLimit(
            hipLimitDevRuntimePendingLaunchCount, "hipLimitDevRuntimePendingLaunchCount", devRuntimePendingLaunchCount);
      }
    }
    */

    if (verbose_) {
      size_t value;
      log << "ROCm limits\n";
      /*
      hipCheck(hipDeviceGetLimit(&value, hipLimitPrintfFifoSize));
      log << "  printf buffer size:        " << std::setw(10) << value / (1 << 20) << " MB\n";
      */
      hipCheck(hipDeviceGetLimit(&value, hipLimitStackSize));
      log << "  stack size:                " << std::setw(10) << value / (1 << 10) << " kB\n";
      hipCheck(hipDeviceGetLimit(&value, hipLimitMallocHeapSize));
      log << "  malloc heap size:          " << std::setw(10) << value / (1 << 20) << " MB\n";
      /*
      if ((properties.major > 3) or (properties.major == 3 and properties.minor >= 5)) {
        hipCheck(hipDeviceGetLimit(&value, hipLimitDevRuntimeSyncDepth));
        log << "  runtime sync depth:           " << std::setw(10) << value << '\n';
        hipCheck(hipDeviceGetLimit(&value, hipLimitDevRuntimePendingLaunchCount));
        log << "  runtime pending launch count: " << std::setw(10) << value << '\n';
      }
      */
    }
  }

  edm::Service<edm::ResourceInformation> resourceInformationService;
  if (resourceInformationService.isAvailable()) {
    std::vector<std::string> modelsV(models.begin(), models.end());
    resourceInformationService->setGPUModels(modelsV);
    /*
    std::string nvidiaDriverVersion{systemDriverVersion};
    resourceInformationService->setNvidiaDriverVersion(nvidiaDriverVersion);
    resourceInformationService->setCudaDriverVersion(driverVersion);
    resourceInformationService->setCudaRuntimeVersion(runtimeVersion);
    */
  }

  if (verbose_) {
    log << '\n' << "ROCmService fully initialized";
  }
  enabled_ = true;
}

ROCmService::~ROCmService() {
  if (enabled_) {
    for (int i = 0; i < numberOfDevices_; ++i) {
      hipCheck(hipSetDevice(i));
      hipCheck(hipDeviceSynchronize());
      // Explicitly destroys and cleans up all resources associated with the current device in the
      // current process. Any subsequent API call to this device will reinitialize the device.
      // Useful to check for memory leaks.
      hipCheck(hipDeviceReset());
    }
  }
}

void ROCmService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("enabled", true);
  desc.addUntracked<bool>("verbose", false);

  edm::ParameterSetDescription limits;
  /*
  limits.addUntracked<int>("hipLimitPrintfFifoSize", -1)
      ->setComment("Size in bytes of the shared FIFO used by the printf() device system call.");
  */
  limits.addUntracked<int>("hipLimitStackSize", -1)->setComment("Stack size in bytes of each GPU thread.");
  limits.addUntracked<int>("hipLimitMallocHeapSize", -1)
      ->setComment("Size in bytes of the heap used by the malloc() and free() device system calls.");
  limits.addUntracked<int>("hipLimitDevRuntimeSyncDepth", -1)
      ->setComment("Maximum nesting depth of a grid at which a thread can safely call hipDeviceSynchronize().");
  limits.addUntracked<int>("hipLimitDevRuntimePendingLaunchCount", -1)
      ->setComment("Maximum number of outstanding device runtime launches that can be made from the current device.");
  desc.addUntracked<edm::ParameterSetDescription>("limits", limits)
      ->setComment(
          "See the documentation of hipDeviceSetLimit for more information.\nSetting any of these options to -1 keeps "
          "the default value.");

  descriptions.add("ROCmService", desc);
}

namespace edm {
  namespace service {
    inline bool isProcessWideService(ROCmService const*) { return true; }
  }  // namespace service
}  // namespace edm

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
using ROCmServiceMaker = edm::serviceregistry::ParameterSetMaker<ROCmInterface, ROCmService>;
DEFINE_FWK_SERVICE_MAKER(ROCmService, ROCmServiceMaker);
