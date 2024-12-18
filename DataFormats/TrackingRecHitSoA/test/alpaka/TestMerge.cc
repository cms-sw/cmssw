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
      uint32_t nHitsOne = 200;
      uint32_t nHitsTwo = 10;
    //   int32_t offset = 100;
      uint32_t nModulesOne = 20;
      uint32_t nModulesTwo = 50;

      ::reco::TrackingRecHitHost hostOne(cms::alpakatools::host(), nHitsOne, nModulesOne + 1);
      ::reco::TrackingRecHitHost hostTwo(cms::alpakatools::host(), nHitsTwo, nModulesTwo + 1);
      ::reco::TrackingRecHitHost hostThree(cms::alpakatools::host(), nHitsOne + nHitsTwo, nModulesOne + nModulesTwo + 1);
      
      auto hitOneView = hostOne.view<::reco::TrackingRecHitSoA>();
      auto hitTwoView = hostTwo.view<::reco::TrackingRecHitSoA>();
      
      auto hitOneModuleView = hostOne.view<::reco::HitModuleSoA>();
      auto hitTwoModuleView = hostTwo.view<::reco::HitModuleSoA>();


      for (uint32_t i = 0; i < nModulesOne + 1; ++i)
        hitOneModuleView[i].moduleStart() = i * 2;
      for (uint32_t i = 0; i < nModulesTwo + 1; ++i)
        hitTwoModuleView[i].moduleStart() = i * 3;
      for (uint32_t i = 0; i < nHitsOne; ++i)
        hitOneView[i].xGlobal() = 2.f;
      for (uint32_t i = 0; i < nHitsTwo; ++i)
        hitTwoView[i].xGlobal() = i + 2;
      
      
      auto hitThreeView = hostThree.view<::reco::TrackingRecHitSoA>();
      auto hitThreeModuleView = hostThree.view<::reco::HitModuleSoA>();
      
      alpaka::memcpy(queue, hostThree.buffer(), hostOne.buffer(), alpaka::getExtentProduct(hostOne.buffer()));
    //   alpaka::memcpy(queue, hostThree.buffer()[nHitsOne], hostTwo.buffer());
      alpaka::wait(queue);

      for (uint32_t i = 0; i < nHitsOne + nHitsTwo; ++i)
        std::cout <<  i << " - " << hitThreeView[i].xGlobal() << std::endl;
    //   for (uint32_t i = 0; i < nHitsOne + 10; ++i)
    //     std::cout << hostThree.buffer().data()[i] << std::endl;

      
    }
  }

  return EXIT_SUCCESS;
}
