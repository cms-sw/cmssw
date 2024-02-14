#include <alpaka/alpaka.hpp>
#include <unistd.h>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigiErrorsSoACollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testDigisSoA {

    void runKernels(SiPixelDigiErrorsSoAView digiErrors_view, Queue& queue);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

int main() {
  const auto host = cms::alpakatools::host();
  const auto device = cms::alpakatools::devices<Platform>()[0];
  Queue queue(device);

  // Inner scope to deallocate memory before destroying the stream
  {
    // Instantiate tracks on device. PortableDeviceCollection allocates
    // SoA on device automatically.
    SiPixelDigiErrorsSoACollection digiErrors_d(1000, queue);
    testDigisSoA::runKernels(digiErrors_d.view(), queue);

    // Instantate tracks on host. This is where the data will be
    // copied to from device.
    SiPixelDigiErrorsHost digiErrors_h(digiErrors_d.view().metadata().size(), queue);
    alpaka::memcpy(queue, digiErrors_h.buffer(), digiErrors_d.const_buffer());
    std::cout << "digiErrors_h.view().metadata().size(): " << digiErrors_h.view().metadata().size() << std::endl;
    std::cout << "digiErrors_h.view()[100].pixelErrors().rawId: " << digiErrors_h.view()[100].pixelErrors().rawId
              << std::endl;
    std::cout << "digiErrors_h.view()[100].pixelErrors().word: " << digiErrors_h.view()[100].pixelErrors().word
              << std::endl;
    std::cout << "digiErrors_h.view()[100].pixelErrors().errorType: "
              << digiErrors_h.view()[100].pixelErrors().errorType << std::endl;
    std::cout << "digiErrors_h.view()[100].pixelErrors().fedId: " << digiErrors_h.view()[100].pixelErrors().fedId
              << std::endl;
    alpaka::wait(queue);
  }

  return 0;
}
