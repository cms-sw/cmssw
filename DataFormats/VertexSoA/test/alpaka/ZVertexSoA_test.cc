/**
   Simple test for the reco::ZVertexSoA data structure
   which inherits from Portable{Host}Collection.

   Creates an instance of the class (automatically allocates
   memory on device), passes the view of the SoA data to
   the kernels which:
   - Fill the SoA with data.
   - Verify that the data written is correct.

   Then, the SoA data are copied back to Host, where
   a temporary host-side view (tmp_view) is created using
   the same Layout to access the data on host and print it.
 */

#include <cstdlib>
#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "ZVertexSoA_test.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// Run 3 values, used for testing
constexpr uint32_t maxTracks = 32 * 1024;
constexpr uint32_t maxVertices = 1024;

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

    // Inner scope to deallocate memory before destroying the stream
    {
      // Instantiate vertices on device. PortableCollection allocates
      // SoA on device automatically.
      ZVertexSoACollection zvertex_d({{maxTracks, maxVertices}}, queue);
      testZVertexSoAT::runKernels(zvertex_d.view(), zvertex_d.view<reco::ZVertexTracksSoA>(), queue);

      // If the device is actually the host, use the collection as-is.
      // Otherwise, copy the data from the device to the host.
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
      ZVertexHost zvertex_h = std::move(zvertex_d);
#else
      ZVertexHost zvertex_h = cms::alpakatools::CopyToHost<ZVertexSoACollection>::copyAsync(queue, zvertex_d);
#endif
      alpaka::wait(queue);
      std::cout << zvertex_h.view().metadata().size() << std::endl;

      // Print results
      std::cout << "idv\t"
                << "zv\t"
                << "wv\t"
                << "chi2\t"
                << "ptv2\t"
                << "ndof\t"
                << "sortInd\t"
                << "nvFinal\n";

      auto vtx_v = zvertex_h.view<reco::ZVertexSoA>();
      auto trk_v = zvertex_h.view<reco::ZVertexTracksSoA>();
      for (int i = 0; i < 10; ++i) {
        auto vi = vtx_v[i];
        auto ti = trk_v[i];
        std::cout << (int)ti.idv() << "\t" << vi.zv() << "\t" << vi.wv() << "\t" << vi.chi2() << "\t" << vi.ptv2()
                  << "\t" << (int)ti.ndof() << "\t" << vi.sortInd() << "\t" << (int)vtx_v.nvFinal() << std::endl;
      }
    }
  }

  return EXIT_SUCCESS;
}
