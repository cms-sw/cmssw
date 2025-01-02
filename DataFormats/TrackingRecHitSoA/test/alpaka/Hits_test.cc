#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

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
      auto moduleStartH =
          cms::alpakatools::make_host_buffer<uint32_t[]>(queue, pixelTopology::Phase1::numberOfModules + 1);
      for (size_t i = 0; i < pixelTopology::Phase1::numberOfModules + 1; ++i) {
        moduleStartH[i] = i * 2;
      }
      auto moduleStartD =
          cms::alpakatools::make_device_buffer<uint32_t[]>(queue, pixelTopology::Phase1::numberOfModules + 1);
      alpaka::memcpy(queue, moduleStartD, moduleStartH);
      TrackingRecHitsSoACollection<pixelTopology::Phase1> tkhit(queue, nHits, offset, moduleStartD.data());

      testTrackingRecHitSoA::runKernels<pixelTopology::Phase1>(tkhit.view(), queue);
      tkhit.updateFromDevice(queue);

#if defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED or defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
      // requires c++23 to make cms::alpakatools::CopyToHost compile using if constexpr
      // see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
      TrackingRecHitHost<pixelTopology::Phase1> const& host_collection = tkhit;
      // wait for the kernel to complete
      alpaka::wait(queue);
#else
      using CopyT = cms::alpakatools::CopyToHost<TrackingRecHitDevice<pixelTopology::Phase1, Device> >;
      TrackingRecHitHost<pixelTopology::Phase1> host_collection = CopyT::copyAsync(queue, tkhit);
      // wait for the kernel and the copy to complete
      alpaka::wait(queue);
      CopyT::postCopy(host_collection);
#endif

      assert(tkhit.nHits() == nHits);
      assert(tkhit.offsetBPIX2() == 22);  // set in the kernel
      assert(tkhit.nHits() == host_collection.nHits());
      assert(tkhit.offsetBPIX2() == host_collection.offsetBPIX2());
    }
  }

  return EXIT_SUCCESS;
}
