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
      constexpr float constYG = -14.0458;
      constexpr float constYL = -27.0855;

      SiPixelClustersSoACollection clusters(queue, nModules);
      clusters.setNClusters(nHits, offset);

      testTrackingRecHitSoA::initializeClusters(queue, clusters.view());

      TrackingRecHitsSoACollection tkhit(queue, clusters);
      testTrackingRecHitSoA::initializeModules(queue, tkhit.view().hitModules(), clusters.view());
      testTrackingRecHitSoA::initializeTracks(queue, tkhit.view().trackingHits());

      testTrackingRecHitSoA::runKernels(queue, tkhit.view());
      tkhit.updateFromDevice(queue);

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED or defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
      // requires c++23 to make cms::alpakatools::CopyToHost compile using if constexpr
      // see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
      ::reco::TrackingRecHitHost const& host_collection = tkhit;
#else
      ::reco::TrackingRecHitHost host_collection =
          cms::alpakatools::CopyToHost< ::reco::TrackingRecHitDevice<Device> >::copyAsync(queue, tkhit);
      alpaka::wait(queue);
#endif

      alpaka::QueueCpuBlocking queue_host{cms::alpakatools::host()};

      ::reco::TrackingRecHitHost host_collection_2(cms::alpakatools::host(), nHits, nModules);

      for (uint32_t i = 0; i < nHits; ++i) {
        host_collection_2.view().trackingHits()[i].yLocal() = constYL;
      }
      assert(host_collection_2.view().trackingHits().yLocal()[nHits - 1] == constYL);

      // wait for the copy above to complete
      alpaka::wait(queue_host);

      assert(host_collection.view().trackingHits().xLocal()[12] == 24.);
      assert(host_collection.view().trackingHits().yGlobal()[int(nHits / 2)] == constYG);
      assert(static_cast<uint32_t>(host_collection.view().hitModules().metadata().size()) == nModules + 1);
      assert(host_collection.nModules() == nModules);

      for (uint32_t i = 0; i < static_cast<uint32_t>(host_collection.view().hitModules().metadata().size()); ++i) {
        assert(host_collection.view().hitModules()[i].moduleStart() == 2 * i);
      }

      assert(tkhit.nHits() == nHits);
      assert(tkhit.offsetBPIX2() == 22);  // set in the kernel
      assert(tkhit.nHits() == host_collection.nHits());
      assert(tkhit.offsetBPIX2() == host_collection.offsetBPIX2());
    }
  }

  return EXIT_SUCCESS;
}
