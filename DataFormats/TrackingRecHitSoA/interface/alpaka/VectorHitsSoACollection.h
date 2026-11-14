#ifndef DataFormats_TrackingRecHitSoA_interface_alpaka_VectorHitsSoACollection_h
#define DataFormats_TrackingRecHitSoA_interface_alpaka_VectorHitsSoACollection_h

// #define GPU_DEBUG

#include <cstdint>
#include <cstring>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using VectorHitsSoACollection = std::
      conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ::reco::VectorHitsHost, ::reco::VectorHitsDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

namespace cms::alpakatools {

  template <typename TDevice>
  struct CopyToHost<::reco::VectorHitsDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ::reco::VectorHitsDevice<TDevice> const& deviceData) {
      auto nVectorHits = deviceData.nVectorHits();

      reco::VectorHitsHost hostData(queue, nVectorHits, deviceData.nModules());

      if (nVectorHits == 0) {
        std::memset(hostData.buffer().data(),
                    0,
                    alpaka::getExtentProduct(hostData.buffer()) * sizeof(alpaka::Elem<reco::VectorHitsHost::Buffer>));
        return hostData;
      }

      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());

#ifdef GPU_DEBUG
      printf("VectorHitsSoACollection: I'm copying to host.\n");
      alpaka::wait(queue);
      assert(deviceData.nVectorHits() == hostData.nVectorHits());
      assert(deviceData.nModules() == hostData.nModules());
#endif

      return hostData;
    }
  };

  template <>
  struct CopyToDevice<::reco::VectorHitsHost> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, reco::VectorHitsHost const& hostData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;

      auto nVectorHits = hostData.nVectorHits();

      reco::VectorHitsDevice<TDevice> deviceData(queue, nVectorHits, hostData.nModules());

      if (nVectorHits == 0) {
        // alpaka::memset takes an lvalue reference to the buffer
        auto buffer = deviceData.buffer();
        alpaka::memset(queue, buffer, 0);
        return deviceData;
      }

      alpaka::memcpy(queue, deviceData.buffer(), hostData.buffer());

#ifdef GPU_DEBUG
      printf("VectorHitsSoACollection: I'm copying to device.\n");
      alpaka::wait(queue);
      assert(deviceData.nVectorHits() == hostData.nVectorHits());
      assert(deviceData.nModules() == hostData.nModules());
#endif
      return deviceData;
    }
  };

}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::VectorHitsSoACollection, reco::VectorHitsHost);

#endif  // DataFormats_TrackingRecHitSoA_interface_alpaka_VectorHitsSoACollection_h
