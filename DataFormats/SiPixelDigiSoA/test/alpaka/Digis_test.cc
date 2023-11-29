#include <unistd.h>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoAv2.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  namespace testDigisSoA {

    void runKernels(SiPixelDigisSoAv2View digis_view, Queue& queue);

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
    SiPixelDigisCollection digis_d(1000, queue);
    testDigisSoA::runKernels(digis_d.view(), queue);

    // Instantate tracks on host. This is where the data will be
    // copied to from device.
    SiPixelDigisHost digis_h(digis_d.view().metadata().size(), queue);

    std::cout << digis_h.view().metadata().size() << std::endl;
    alpaka::memcpy(queue, digis_h.buffer(), digis_d.const_buffer());
    alpaka::wait(queue);
  }

  return 0;
}
