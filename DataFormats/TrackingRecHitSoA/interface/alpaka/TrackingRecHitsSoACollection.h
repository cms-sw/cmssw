#ifndef DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsSoACollection_h
#define DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsSoACollection_h

// #define GPU_DEBUG

#include <cstdint>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using TrackingRecHitsSoACollection = std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>,
                                                          ::reco::TrackingRecHitHost,
                                                          ::reco::TrackingRecHitDevice<Device>>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

namespace cms::alpakatools {
  template <typename TDevice>
  struct CopyToHost<::reco::TrackingRecHitDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ::reco::TrackingRecHitDevice<TDevice> const& deviceData) {
      auto nHits = deviceData.nHits();

      reco::TrackingRecHitHost hostData(queue, nHits, deviceData.nModules());

      // Don't bother if zero hits
      if (nHits == 0) {
        std::memset(
            hostData.buffer().data(),
            0,
            alpaka::getExtentProduct(hostData.buffer()) * sizeof(alpaka::Elem<reco::TrackingRecHitHost::Buffer>));
        return hostData;
      }

      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());
#ifdef GPU_DEBUG
      printf("TrackingRecHitsSoACollection: I'm copying to host.\n");
      alpaka::wait(queue);
      assert(deviceData.nHits() == hostData.nHits());
      assert(deviceData.nModules() == hostData.nModules());
      assert(deviceData.offsetBPIX2() == hostData.offsetBPIX2());
#endif

      return hostData;
    }
  };

  template <>
  struct CopyToDevice<::reco::TrackingRecHitHost> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, reco::TrackingRecHitHost const& hostData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;

      auto nHits = hostData.nHits();

      reco::TrackingRecHitDevice<TDevice> deviceData(queue, nHits, hostData.nModules());

      if (nHits == 0) {
        std::memset(
            deviceData.buffer().data(),
            0,
            alpaka::getExtentProduct(deviceData.buffer()) * sizeof(alpaka::Elem<reco::TrackingRecHitHost::Buffer>));
        return deviceData;
      }

      alpaka::memcpy(queue, deviceData.buffer(), hostData.buffer());

#ifdef GPU_DEBUG
      printf("TrackingRecHitsSoACollection: I'm copying to device.\n");
      alpaka::wait(queue);
      assert(deviceData.nHits() == hostData.nHits());
      assert(deviceData.nModules() == hostData.nModules());
      assert(deviceData.offsetBPIX2() == hostData.offsetBPIX2());
#endif
      return deviceData;
    }
  };

}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::TrackingRecHitsSoACollection, reco::TrackingRecHitHost);

#endif  // DataFormats_TrackingRecHitSoA_interface_alpaka_TrackingRecHitsSoACollection_h
