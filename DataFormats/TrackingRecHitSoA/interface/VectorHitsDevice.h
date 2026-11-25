#ifndef DataFormats_TrackingRecHitSoA_interface_VectorHitsDevice_h
#define DataFormats_TrackingRecHitSoA_interface_VectorHitsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  template <typename TDev>
  using VectorHitPortableCollectionDevice = PortableDeviceCollection<TDev, reco::VectorHitBlocksSoA>;

  template <typename TDev>
  class VectorHitsDevice : public VectorHitPortableCollectionDevice<TDev> {
  public:
    VectorHitsDevice() = default;

    VectorHitsDevice(edm::Uninitialized) : VectorHitPortableCollectionDevice<TDev>{edm::kUninitialized} {}

    // The module block is allocated with nModules+1 elements and carries no payload: its extent is
    // the only record of the module count.
    template <typename TQueue>
    explicit VectorHitsDevice(TQueue queue, uint32_t nVectorHits, uint32_t nModules)
        : VectorHitPortableCollectionDevice<TDev>(queue, nVectorHits, nModules + 1) {}

    // Number of vector hits in the collection
    uint32_t nVectorHits() const { return this->view().vectorHits().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().vectorHitModules().metadata().size() - 1; }
  };

}  // namespace reco

namespace ngt {

  template <typename TDev>
  struct MemoryCopyTraits<reco::VectorHitsDevice<TDev>> {
    using value_type = reco::VectorHitsDevice<TDev>;

    struct Properties {
      uint32_t nVectorHits;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nVectorHits(), object.nModules()}; }

    template <typename TQueue>
      requires(alpaka::concepts::Queue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in device global memory.
      object = value_type(queue, prop.nVectorHits, prop.nModules);
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

#endif  // DataFormats_TrackingRecHitSoA_interface_VectorHitsDevice_h
