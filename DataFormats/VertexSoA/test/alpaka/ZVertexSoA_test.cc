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

#include "DataFormats/VertexSoA/interface/ZVertexDevice.h"
#include "DataFormats/VertexSoA/interface/ZVertexHost.h"
#include "DataFormats/VertexSoA/interface/alpaka/ZVertexSoACollection.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "ZVertexSoA_test.h"

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

    // Inner scope to deallocate memory before destroying the stream
    {
      // Instantiate vertices on device. PortableCollection allocates
      // SoA on device automatically.
      ZVertexSoACollection zvertex_d(queue);
      testZVertexSoAT::runKernels(zvertex_d.view(), queue);

      // Instantate vertices on host. This is where the data will be
      // copied to from device.
      ZVertexHost zvertex_h(queue);
      std::cout << zvertex_h.view().metadata().size() << std::endl;
      alpaka::memcpy(queue, zvertex_h.buffer(), zvertex_d.const_buffer());
      alpaka::wait(queue);

      // Print results
      std::cout << "idv\t"
                << "zv\t"
                << "wv\t"
                << "chi2\t"
                << "ptv2\t"
                << "ndof\t"
                << "sortInd\t"
                << "nvFinal\n";

      for (int i = 0; i < 10; ++i) {
        std::cout << (int)zvertex_h.view()[i].idv() << '\t' << zvertex_h.view()[i].zv() << '\t'
                  << zvertex_h.view()[i].wv() << '\t' << zvertex_h.view()[i].chi2() << '\t'
                  << zvertex_h.view()[i].ptv2() << '\t' << (int)zvertex_h.view()[i].ndof() << '\t'
                  << (int)zvertex_h.view()[i].sortInd() << '\t' << (int)zvertex_h.view().nvFinal() << '\n';
      }
    }
  }

  return EXIT_SUCCESS;
}
