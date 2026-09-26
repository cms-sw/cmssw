#ifndef DataFormats_TrackingRecHitSoA_interface_alpaka_OTRecHitsSoACollection_h
#define DataFormats_TrackingRecHitSoA_interface_alpaka_OTRecHitsSoACollection_h

#include <cstdint>
#include <cstring>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  // OTRecHitsSoACollection is either OTRecHitsHost (for CPU) or OTRecHitsDevice<Device> (for GPU)
  using OTRecHitsSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ::reco::OTRecHitsHost, ::reco::OTRecHitsDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

namespace cms::alpakatools {

  // Copy from device to host
  template <typename TDevice>
  struct CopyToHost<::reco::OTRecHitsDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ::reco::OTRecHitsDevice<TDevice> const& deviceData) {
      auto nHits = deviceData.nHits();

      ::reco::OTRecHitsHost hostData(queue, nHits, deviceData.nModules());

      // Don't bother if zero hits
      if (nHits == 0) {
        std::memset(hostData.buffer().data(),
                    0,
                    alpaka::getExtentProduct(hostData.buffer()) * sizeof(alpaka::Elem<::reco::OTRecHitsHost::Buffer>));
        return hostData;
      }

      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());

      return hostData;
    }
  };

  // Copy from host to device
  template <>
  struct CopyToDevice<::reco::OTRecHitsHost> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, ::reco::OTRecHitsHost const& hostData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;

      auto nHits = hostData.nHits();

      ::reco::OTRecHitsDevice<TDevice> deviceData(queue, nHits, hostData.nModules());

      if (nHits == 0) {
        // Use alpaka::memset for device memory instead of std::memset
        // Need to store buffer in a variable since alpaka::memset requires an lvalue reference
        auto buffer = deviceData.buffer();
        alpaka::memset(queue, buffer, 0);
        return deviceData;
      }

      alpaka::memcpy(queue, deviceData.buffer(), hostData.buffer());

      return deviceData;
    }
  };

}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::OTRecHitsSoACollection, reco::OTRecHitsHost);

#endif  // DataFormats_TrackingRecHitSoA_interface_alpaka_OTRecHitsSoACollection_h
