#ifndef DataFormats_TrackingRecHitSoA_interface_alpaka_StubsSoACollection_h
#define DataFormats_TrackingRecHitSoA_interface_alpaka_StubsSoACollection_h

// #define GPU_DEBUG

#include <cstdint>
#include <cstring>
#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/concepts.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using StubsSoACollection =
      std::conditional_t<std::is_same_v<Device, alpaka::DevCpu>, ::reco::StubsHost, ::reco::StubsDevice<Device>>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

namespace cms::alpakatools {

  template <typename TDevice>
  struct CopyToHost<::reco::StubsDevice<TDevice>> {
    template <typename TQueue>
    static auto copyAsync(TQueue& queue, ::reco::StubsDevice<TDevice> const& deviceData) {
      auto nStubs = deviceData.nStubs();

      reco::StubsHost hostData(queue, nStubs, deviceData.nModules());

      if (nStubs == 0) {
        std::memset(hostData.buffer().data(),
                    0,
                    alpaka::getExtentProduct(hostData.buffer()) * sizeof(alpaka::Elem<reco::StubsHost::Buffer>));
        return hostData;
      }

      alpaka::memcpy(queue, hostData.buffer(), deviceData.buffer());

#ifdef GPU_DEBUG
      printf("StubsSoACollection: I'm copying to host.\n");
      alpaka::wait(queue);
      assert(deviceData.nStubs() == hostData.nStubs());
      assert(deviceData.nModules() == hostData.nModules());
#endif

      return hostData;
    }
  };

  template <>
  struct CopyToDevice<::reco::StubsHost> {
    template <cms::alpakatools::NonCPUQueue TQueue>
    static auto copyAsync(TQueue& queue, reco::StubsHost const& hostData) {
      using TDevice = typename alpaka::trait::DevType<TQueue>::type;

      auto nStubs = hostData.nStubs();

      reco::StubsDevice<TDevice> deviceData(queue, nStubs, hostData.nModules());

      if (nStubs == 0) {
        // alpaka::memset takes an lvalue reference to the buffer
        auto buffer = deviceData.buffer();
        alpaka::memset(queue, buffer, 0);
        return deviceData;
      }

      alpaka::memcpy(queue, deviceData.buffer(), hostData.buffer());

#ifdef GPU_DEBUG
      printf("StubsSoACollection: I'm copying to device.\n");
      alpaka::wait(queue);
      assert(deviceData.nStubs() == hostData.nStubs());
      assert(deviceData.nModules() == hostData.nModules());
#endif
      return deviceData;
    }
  };

}  // namespace cms::alpakatools

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(reco::StubsSoACollection, reco::StubsHost);

#endif  // DataFormats_TrackingRecHitSoA_interface_alpaka_StubsSoACollection_h
