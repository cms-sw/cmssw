#ifndef DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h
#define DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"

namespace reco {

  template <typename TDev>
  using OTRecHitPortableCollectionDevice = PortableDeviceCollection<TDev, reco::OTRecHitBlocksSoA>;

  template <typename TDev>
  class OTRecHitsDevice : public OTRecHitPortableCollectionDevice<TDev> {
  public:
    OTRecHitsDevice() = default;

    OTRecHitsDevice(edm::Uninitialized) : OTRecHitPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // nHits, nModules: SoA sizes (nModules+1 elements for the cumulative-sum array).
    template <typename TQueue>
    explicit OTRecHitsDevice(TQueue queue, uint32_t nHits, uint32_t nModules)
        : OTRecHitPortableCollectionDevice<TDev>(queue, nHits, nModules + 1) {}

    uint32_t nHits() const { return this->view().otRecHits().metadata().size(); }

    uint32_t nModules() const { return this->view().otHitModules().metadata().size() - 1; }

    // No-op: the data is already on the device.
    template <typename TQueue>
    void updateFromDevice(TQueue) {}
  };

}  // namespace reco

namespace ngt {

  template <typename TDev>
  struct MemoryCopyTraits<reco::OTRecHitsDevice<TDev>> {
    using value_type = reco::OTRecHitsDevice<TDev>;

    struct Properties {
      uint32_t nHits;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nHits(), object.nModules()}; }

    template <typename TQueue>
      requires(alpaka::concepts::Queue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in device global memory.
      object = value_type(queue, prop.nHits, prop.nModules);
    }

    static std::vector<std::span<std::byte>> regions(value_type& object) {
      std::byte* address = reinterpret_cast<std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }

    static std::vector<std::span<const std::byte>> regions(value_type const& object) {
      const std::byte* address = reinterpret_cast<const std::byte*>(object.buffer().data());
      size_t size = alpaka::getExtentProduct(object.buffer());
      return {{address, size}};
    }
  };

}  // namespace ngt

#endif  // DataFormats_TrackingRecHitSoA_interface_OTRecHitsDevice_h
