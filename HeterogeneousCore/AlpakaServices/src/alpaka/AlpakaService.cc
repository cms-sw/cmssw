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
#include "HeterogeneousCore/CUDAServices/interface/CUDAInterface.h"
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/ROCmServices/interface/ROCmInterface.h"
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

namespace {

  // Note: we cannot use "uint64_t" with the ParameterSet-related functions, because the template specialisations expect "unsigned long long", while "uint64_t" expands to "unsigned long".

  edm::ParameterSetDescription createAllocatorConfig(
      cms::alpakatools::AllocatorConfig const& alloc = cms::alpakatools::AllocatorConfig{}) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<uint32_t>("binGrowth", alloc.binGrowth)
        ->setComment("Bin growth factor (bin_growth in cub::CachingDeviceAllocator)");
    desc.addUntracked<uint32_t>("minBin", alloc.minBin)
        ->setComment(
            "Smallest bin, corresponds to binGrowth^minBin bytes (min_bin in cub::CachingDeviceAllocator).\n8 "
            "corresponds to 256 bytes.");
    desc.addUntracked<uint32_t>("maxBin", alloc.maxBin)
        ->setComment(
            "Largest bin, corresponds to binGrowth^maxBin bytes (max_bin in cub::CachingDeviceAllocator).\n30 "
            "corresponds do 1 GiB.\nNote that unlike in cub, allocations larger than binGrowth^maxBin are set to "
            "fail.");
    desc.addUntracked<unsigned long long>("maxCachedBytes", alloc.maxCachedBytes)
        ->setComment("Total storage for the allocator; 0 means no limit.");
    desc.addUntracked<double>("maxCachedFraction", alloc.maxCachedFraction)
        ->setComment(
            "Fraction of total device memory taken for the allocator; 0 means no limit.\nIf both maxCachedBytes and "
            "maxCachedFraction are non-zero, the smallest resulting value is used.");
    desc.addUntracked<bool>("fillAllocations", alloc.fillAllocations)
        ->setComment("Fill all newly allocated or re-used memory blocks with fillAllocationValue.");
    desc.addUntracked<uint32_t>("fillAllocationValue", alloc.fillAllocationValue)
        ->setComment("Byte value used to fill all newly allocated or re-used memory blocks");
    desc.addUntracked<bool>("fillReallocations", alloc.fillReallocations)
        ->setComment(
            "Fill only the re-used memory blocks with fillReallocationValue.\nIf both fillAllocations and "
            "fillReallocations are true, fillAllocationValue is used for newly allocated blocks and "
            "fillReallocationValue is used for re-allocated blocks.");
    desc.addUntracked<uint32_t>("fillReallocationValue", alloc.fillReallocationValue)
        ->setComment("Byte value used to fill all re-used memory blocks");
    desc.addUntracked<bool>("fillDeallocations", alloc.fillDeallocations)
        ->setComment("Fill memory blocks with fillDeallocationValue before freeing or caching them for re-use");
    desc.addUntracked<uint32_t>("fillDeallocationValue", alloc.fillDeallocationValue)
        ->setComment("Byte value used to fill all deallocated or cached memory blocks");
    desc.addUntracked<bool>("fillCaches", alloc.fillCaches)
        ->setComment(
            "Fill memory blocks with fillCacheValue before caching them for re-use.\nIf both fillDeallocations and "
            "fillCaches are true, fillDeallocationValue is used for blocks about to be freed and fillCacheValue is "
            "used for blocks about to be cached.");
    desc.addUntracked<uint32_t>("fillCacheValue", alloc.fillCacheValue)
        ->setComment("Byte value used to fill all cached memory blocks");
    return desc;
  }

  cms::alpakatools::AllocatorConfig parseAllocatorConfig(edm::ParameterSet const& config) {
    cms::alpakatools::AllocatorConfig alloc;
    alloc.binGrowth = config.getUntrackedParameter<uint32_t>("binGrowth");
    alloc.minBin = config.getUntrackedParameter<uint32_t>("minBin");
    alloc.maxBin = config.getUntrackedParameter<uint32_t>("maxBin");
    alloc.maxCachedBytes = config.getUntrackedParameter<unsigned long long>("maxCachedBytes");
    alloc.maxCachedFraction = config.getUntrackedParameter<double>("maxCachedFraction");
    alloc.fillAllocations = config.getUntrackedParameter<bool>("fillAllocations");
    alloc.fillAllocationValue = static_cast<uint8_t>(config.getUntrackedParameter<uint32_t>("fillAllocationValue"));
    alloc.fillReallocations = config.getUntrackedParameter<bool>("fillReallocations");
    alloc.fillReallocationValue = static_cast<uint8_t>(config.getUntrackedParameter<uint32_t>("fillReallocationValue"));
    alloc.fillDeallocations = config.getUntrackedParameter<bool>("fillDeallocations");
    alloc.fillDeallocationValue = static_cast<uint8_t>(config.getUntrackedParameter<uint32_t>("fillDeallocationValue"));
    alloc.fillCaches = config.getUntrackedParameter<bool>("fillCaches");
    alloc.fillCacheValue = static_cast<uint8_t>(config.getUntrackedParameter<uint32_t>("fillCacheValue"));
    return alloc;
  }

}  // namespace

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  AlpakaService::AlpakaService(edm::ParameterSet const& config, edm::ActivityRegistry&)
      : enabled_(config.getUntrackedParameter<bool>("enabled")),
        verbose_(config.getUntrackedParameter<bool>("verbose")) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // rely on the CUDAService to initialise the CUDA devices
    edm::Service<CUDAInterface> cuda;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    // rely on the ROCmService to initialise the ROCm devices
    edm::Service<ROCmInterface> rocm;
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

    // TODO from Andrea Bocci:
    //   - extract and print more information about the platform and devices

    if (not enabled_) {
      edm::LogInfo("AlpakaService") << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " disabled by configuration";
      return;
    }

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    if (not cuda or not cuda->enabled()) {
      enabled_ = false;
      edm::LogInfo("AlpakaService") << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " disabled by CUDAService";
      return;
    }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    if (not rocm or not rocm->enabled()) {
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
    cms::alpakatools::AllocatorConfig hostAllocatorConfig =
        parseAllocatorConfig(config.getUntrackedParameter<edm::ParameterSet>("hostAllocator"));
    cms::alpakatools::getHostCachingAllocator<Queue>(hostAllocatorConfig, verbose_);
    cms::alpakatools::AllocatorConfig deviceAllocatorConfig =
        parseAllocatorConfig(config.getUntrackedParameter<edm::ParameterSet>("deviceAllocator"));
    for (auto const& device : devices)
      cms::alpakatools::getDeviceCachingAllocator<Device, Queue>(device, deviceAllocatorConfig, verbose_);
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
    desc.addUntracked<edm::ParameterSetDescription>("hostAllocator", createAllocatorConfig())
        ->setComment("Configuration for the host's CachingAllocator");
    desc.addUntracked<edm::ParameterSetDescription>("deviceAllocator", createAllocatorConfig())
        ->setComment("Configuration for the devices' CachingAllocator");

    descriptions.add(ALPAKA_TYPE_ALIAS_NAME(AlpakaService), desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
