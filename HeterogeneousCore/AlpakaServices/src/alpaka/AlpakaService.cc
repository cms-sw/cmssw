#include <boost/core/demangle.hpp>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/EventCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/getDeviceCachingAllocator.h"
#include "HeterogeneousCore/AlpakaInterface/interface/getHostCachingAllocator.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmService.h"
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  AlpakaService::AlpakaService(edm::ParameterSet const& config, edm::ActivityRegistry&)
      : enabled_(config.getUntrackedParameter<bool>("enabled")),
        verbose_(config.getUntrackedParameter<bool>("verbose")) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // rely on the CUDAService to initialise the CUDA devices
    edm::Service<CUDAService> cudaService;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // rely on the ROCmService to initialise the ROCm devices
    edm::Service<ROCmService> rocmService;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

    // TODO from Andrea Bocci:
    //   - handle alpaka caching allocators ?
    //   - extract and print more information about the platform and devices

    if (not enabled_) {
      edm::LogInfo("AlpakaService") << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " disabled by configuration";
      return;
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if (not cudaService->enabled()) {
      enabled_ = false;
      edm::LogInfo("AlpakaService") << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " disabled by CUDAService";
      return;
    }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    if (not rocmService->enabled()) {
      enabled_ = false;
      edm::LogInfo("AlpakaService") << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " disabled by ROCmService";
      return;
    }
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

    // enumerate all devices on this platform
    auto const& devices = cms::alpakatools::devices<Platform>();
    if (devices.empty()) {
      const std::string platform = boost::core::demangle(typeid(Platform).name());
      edm::LogWarning("AlpakaService") << "Could not find any devices on platform " << platform << ".\n"
                                       << "Disabling " << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << ".";
      enabled_ = false;
      return;
    }

    {
      const char* suffix[] = {"s.", ":", "s:"};
      const auto n = devices.size();
      edm::LogInfo out("AlpakaService");
      out << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " succesfully initialised.\n";
      out << "Found " << n << " device" << suffix[n < 2 ? n : 2];
      for (auto const& device : devices) {
        out << "\n  - " << alpaka::getName(device);
      }
    }

    // initialise the queue and event caches
    cms::alpakatools::getQueueCache<Queue>().clear();
    cms::alpakatools::getEventCache<Event>().clear();

    // initialise the caching memory allocators
    cms::alpakatools::getHostCachingAllocator<Queue>();
    for (auto const& device : devices)
      cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device);
  }

  AlpakaService::~AlpakaService() {
    // clean up the caching memory allocators
    cms::alpakatools::getHostCachingAllocator<Queue>().freeAllCached();
    for (auto const& device : cms::alpakatools::devices<Platform>())
      cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device).freeAllCached();

    // clean up the queue and event caches
    cms::alpakatools::getQueueCache<Queue>().clear();
    cms::alpakatools::getEventCache<Event>().clear();
  }

  void AlpakaService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("enabled", true);
    desc.addUntracked<bool>("verbose", false);

    descriptions.add(ALPAKA_TYPE_ALIAS_NAME(AlpakaService), desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
