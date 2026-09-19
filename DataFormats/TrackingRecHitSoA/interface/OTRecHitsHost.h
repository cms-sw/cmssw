#ifndef DataFormats_TrackingRecHitSoA_interface_OTRecHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_OTRecHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsSoA.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace reco {

  using OTRecHitPortableCollectionHost = PortableHostCollection<reco::OTRecHitBlocksSoA>;

  class OTRecHitsHost : public OTRecHitPortableCollectionHost {
  public:
    OTRecHitsHost(edm::Uninitialized) : PortableHostCollection<reco::OTRecHitBlocksSoA>{edm::kUninitialized} {}

    // The module block holds nModules+1 elements: it is the cumulative-sum array.
    template <typename TQueue>
    explicit OTRecHitsHost(TQueue queue, uint32_t nHits, uint32_t nModules)
        : OTRecHitPortableCollectionHost(queue, nHits, nModules + 1) {}

    // Number of OT hits in the collection
    uint32_t nHits() const { return this->view().otRecHits().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().otHitModules().metadata().size() - 1; }

    template <typename TQueue>
    void updateFromDevice(TQueue) {}
  };

}  // namespace reco

namespace ngt {

  template <>
  struct MemoryCopyTraits<reco::OTRecHitsHost> {
    using value_type = reco::OTRecHitsHost;

    struct Properties {
      uint32_t nHits;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nHits(), object.nModules()}; }

    static void initialize(value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pageable system memory
      object = value_type(cms::alpakatools::host(), prop.nHits, prop.nModules);
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

#endif  // DataFormats_TrackingRecHitSoA_interface_OTRecHitsHost_h
