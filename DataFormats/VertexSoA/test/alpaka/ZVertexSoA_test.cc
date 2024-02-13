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

#include <alpaka/alpaka.hpp>
#include <unistd.h>
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace std;
using namespace ALPAKA_ACCELERATOR_NAMESPACE;
using namespace reco;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testZVertexSoAT {
    void runKernels(ZVertexSoAView zvertex_view, ZVertexTracksSoAView zvertextracks_view, Queue& queue);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  // Inner scope to deallocate memory before destroying the stream
  {
    // Instantiate vertices on device. PortableCollection allocates
    // SoA on device automatically.
    ZVertexSoACollection zvertex_d(queue);
    testZVertexSoAT::runKernels(zvertex_d.view(), zvertex_d.view<reco::ZVertexTracksSoA>(), queue);

    // Instantate vertices on host. This is where the data will be
    // copied to from device.
    ZVertexHost zvertex_h(queue);
    std::cout << zvertex_h.view().metadata().size() << std::endl;
    alpaka::memcpy(queue, zvertex_h.buffer(), zvertex_d.const_buffer());
    alpaka::wait(queue);

    // Print results
    std::cout << "idv"
              << "\t"
              << "zv"
              << "\t"
              << "wv"
              << "\t"
              << "chi2"
              << "\t"
              << "ptv2"
              << "\t"
              << "ndof"
              << "\t"
              << "sortInd"
              << "\t"
              << "nvFinal" << std::endl;

    auto vtx_v = zvertex_h.view<reco::ZVertexSoA>();
    auto trk_v = zvertex_h.view<reco::ZVertexTracksSoA>();
    for (int i = 0; i < 10; ++i) {
      auto vi = vtx_v[i];
      auto ti = trk_v[i];
      std::cout << (int)ti.idv() << "\t" << vi.zv() << "\t" << vi.wv() << "\t" << vi.chi2() << "\t" << vi.ptv2() << "\t"
                << (int)ti.ndof() << "\t" << vi.sortInd() << "\t" << (int)vtx_v.nvFinal() << std::endl;
    }
  }

  return 0;
}
