#ifndef DataFormats_TrackingRecHitSoA_interface_VectorHitsHost_h
#define DataFormats_TrackingRecHitSoA_interface_VectorHitsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsSoA.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  using VectorHitPortableCollectionHost = PortableHostCollection<reco::VectorHitBlocksSoA>;

  class VectorHitsHost : public VectorHitPortableCollectionHost {
  public:
    VectorHitsHost(edm::Uninitialized) : VectorHitPortableCollectionHost{edm::kUninitialized} {}

    // The module block holds moduleStart with nModules+1 entries, the last one the total
    // number of vector hits.
    template <typename TQueue>
    explicit VectorHitsHost(TQueue queue, uint32_t nVectorHits, uint32_t nModules)
        : VectorHitPortableCollectionHost(queue, nVectorHits, nModules + 1) {}

    // Number of vector hits in the collection
    uint32_t nVectorHits() const { return this->view().vectorHits().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().vectorHitModules().metadata().size() - 1; }
  };

}  // namespace reco

namespace ngt {

  template <>
  struct MemoryCopyTraits<reco::VectorHitsHost> {
    using value_type = reco::VectorHitsHost;

    struct Properties {
      uint32_t nVectorHits;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nVectorHits(), object.nModules()}; }

    static void initialize(value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pageable system memory
      object = value_type(cms::alpakatools::host(), prop.nVectorHits, prop.nModules);
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

#endif  // DataFormats_TrackingRecHitSoA_interface_VectorHitsHost_h
