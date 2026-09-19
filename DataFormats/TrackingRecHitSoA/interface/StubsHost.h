#ifndef DataFormats_TrackingRecHitSoA_interface_StubsHost_h
#define DataFormats_TrackingRecHitSoA_interface_StubsHost_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsSoA.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306

namespace reco {

  using StubPortableCollectionHost = PortableHostCollection<reco::StubBlocksSoA>;

  class StubsHost : public StubPortableCollectionHost {
  public:
    StubsHost(edm::Uninitialized) : PortableHostCollection<reco::StubBlocksSoA>{edm::kUninitialized} {}

    // The module block is allocated with nModules+1 elements and carries no payload: its extent is
    // the only record of the module count.
    template <typename TQueue>
    explicit StubsHost(TQueue queue, uint32_t nStubs, uint32_t nModules)
        : StubPortableCollectionHost(queue, nStubs, nModules + 1) {}

    // Number of stubs in the collection
    uint32_t nStubs() const { return this->view().stubs().metadata().size(); }

    // Number of stacked modules
    uint32_t nModules() const { return this->view().stubModules().metadata().size() - 1; }
  };

}  // namespace reco

namespace ngt {

  template <>
  struct MemoryCopyTraits<reco::StubsHost> {
    using value_type = reco::StubsHost;

    struct Properties {
      uint32_t nStubs;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nStubs(), object.nModules()}; }

    static void initialize(value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pageable system memory
      object = value_type(cms::alpakatools::host(), prop.nStubs, prop.nModules);
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

#endif  // DataFormats_TrackingRecHitSoA_interface_StubsHost_h
