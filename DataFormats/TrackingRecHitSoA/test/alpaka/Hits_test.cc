#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelClusterSoA/interface/alpaka/SiPixelClustersSoACollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"

#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Hits_test.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace ALPAKA_ACCELERATOR_NAMESPACE::reco;

int main() {
  // Get the list of devices on the current platform
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::cerr << "No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend, "
      "the test will be skipped.\n";
    exit(EXIT_FAILURE);
  }

  // Run the test on each device
  for (const auto& device : devices) {
    Queue queue(device);

    // inner scope to deallocate memory before destroying the queue
    {
      uint32_t nHits = 2000;
      int32_t offset = 100;
      uint32_t nModules = 200;

      SiPixelClustersSoACollection clusters(nModules, queue);
      clusters.setNClusters(nHits,offset);

      auto moduleStartH =
          cms::alpakatools::make_host_buffer<uint32_t[]>(queue, nModules + 1);
     
      for (size_t i = 0; i < nModules + 1; ++i) {
        moduleStartH[i] = i * 2;
      }

      auto hitsX = cms::alpakatools::make_host_buffer<float[]>(queue, nHits);
      for (size_t i = 0; i < nHits; ++i) {
        hitsX[i] = float(i) * 2;
      }

      auto moduleStartD = cms::alpakatools::make_device_view<uint32_t>(queue, clusters.view().clusModuleStart(), nHits);
      alpaka::memcpy(queue, moduleStartD, moduleStartH);

      TrackingRecHitsSoACollection tkhit(queue, clusters);
      
      // exercise the copy of a full column (on device)
      auto hitXD = cms::alpakatools::make_device_view<float>(queue, tkhit.view().xLocal(), nHits);
      alpaka::memcpy(queue, hitXD, hitsX);
      
      // exercise the memset of a colum (on device)
      auto hitYD = cms::alpakatools::make_device_view<float>(queue, tkhit.view().yGlobal(), nHits);
      constexpr float constYG = -14.0458;
      std::vector<float> constXV(nHits,constYG);
      auto constYGV_v = cms::alpakatools::make_host_view<float>(constXV.data(),nHits);
      alpaka::memcpy(queue, hitYD, constYGV_v);

      testTrackingRecHitSoA::runKernels(tkhit.view(), tkhit.view<::reco::HitModuleSoA>(), queue);
      tkhit.updateFromDevice(queue);

      


#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED or defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
      // requires c++23 to make cms::alpakatools::CopyToHost compile using if constexpr
      // see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
      ::reco::TrackingRecHitHost const& host_collection = tkhit;
#else
      ::reco::TrackingRecHitHost host_collection =
          cms::alpakatools::CopyToHost<::reco::TrackingRecHitDevice<Device> >::copyAsync(queue, tkhit);
#endif
      
      alpaka::wait(queue);

      alpaka::QueueCpuBlocking queue_host{cms::alpakatools::host()};
    
      ::reco::TrackingRecHitHost host_collection_2(cms::alpakatools::host(), nHits, nModules);

      // exercise the memset of a colum (on host)
      auto hitLYH = cms::alpakatools::make_host_view<float>(host_collection_2.view().yLocal(), nHits);
      constexpr float constYL = -27.0855;
      std::vector<float> constYLV(nHits,constYL);
      auto constYL_v = cms::alpakatools::make_host_view<float>(constYLV.data(),nHits);
      alpaka::memcpy(queue_host, hitLYH, constYL_v);
      // wait for the kernel and the potential copy to complete
      
      assert(host_collection.view().xLocal()[12] == 24.);
      assert(host_collection.view().yGlobal()[int(nHits/2)] == constYG);
      assert(host_collection_2.view().yLocal()[nHits-1] == constYL);

      assert(tkhit.nHits() == nHits);
      assert(tkhit.offsetBPIX2() == 22);  // set in the kernel
      assert(tkhit.nHits() == host_collection.nHits());
      assert(tkhit.offsetBPIX2() == host_collection.offsetBPIX2());
    }
  }

  return EXIT_SUCCESS;
}
