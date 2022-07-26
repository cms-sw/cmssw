#include <boost/core/demangle.hpp>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaServices/interface/alpaka/AlpakaService.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  AlpakaService::AlpakaService(edm::ParameterSet const& config, edm::ActivityRegistry&)
      : enabled_(config.getUntrackedParameter<bool>("enabled")),
        verbose_(config.getUntrackedParameter<bool>("verbose")) {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    // rely on the CUDAService to initialise the CUDA devices
    edm::Service<CUDAService> cudaService;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

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

    // enumerate all devices on this platform
    uint32_t n = alpaka::getDevCount<Platform>();
    if (n == 0) {
      const std::string platform = boost::core::demangle(typeid(Platform).name());
      edm::LogWarning("AlpakaService") << "Could not find any devices on platform " << platform << ".\n"
                                       << "Disabling " << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << ".";
      enabled_ = false;
      return;
    }

    devices_.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      devices_.push_back(alpaka::getDevByIdx<Platform>(i));
      //assert(getDeviceIndex(devices_.back()) == static_cast<int>(i));
    }

    {
      const char* suffix[] = {"s.", ":", "s:"};
      edm::LogInfo out("AlpakaService");
      out << ALPAKA_TYPE_ALIAS_NAME(AlpakaService) << " succesfully initialised.\n";
      out << "Found " << n << " device" << suffix[n < 2 ? n : 2];
      for (auto const& device : devices_) {
        out << "\n  - " << alpaka::getName(device);
      }
    }
  }

  void AlpakaService::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<bool>("enabled", true);
    desc.addUntracked<bool>("verbose", false);

    descriptions.add(ALPAKA_TYPE_ALIAS_NAME(AlpakaService), desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
