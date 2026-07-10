#ifndef DataFormats_SiPixelClusterSoA_interface_SiPixelClustersHost_h
#define DataFormats_SiPixelClusterSoA_interface_SiPixelClustersHost_h

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// TODO: The class is created via inheritance of the PortableCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class SiPixelClustersHost : public PortableHostCollection<SiPixelClustersSoA> {
public:
  explicit SiPixelClustersHost(edm::Uninitialized) : PortableHostCollection<SiPixelClustersSoA>{edm::kUninitialized} {}

  // Constructor for code that does not use alpaka explicitly, using the global "host" object returned by cms::alpakatools::host(), construct the object in pageable system memory.
  explicit SiPixelClustersHost(size_t maxModules)
      : PortableHostCollection<SiPixelClustersSoA>(cms::alpakatools::host(), maxModules + 1) {}

  // Construct the object in pageable system memory.
  explicit SiPixelClustersHost(alpaka_common::DevHost const& host, size_t maxModules)
      : PortableHostCollection<SiPixelClustersSoA>(host, maxModules + 1) {}

  // Construct the object in pinned host memory associated to the given work queue, accessible by the queue's device.
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  explicit SiPixelClustersHost(TQueue queue, size_t maxModules)
      : PortableHostCollection<SiPixelClustersSoA>(queue, maxModules + 1) {}

  void setNClusters(uint32_t nClusters, int32_t offsetBPIX2) {
    nClusters_h = nClusters;
    offsetBPIX2_h = offsetBPIX2;
  }

  uint32_t nClusters() const { return nClusters_h; }
  int32_t offsetBPIX2() const { return offsetBPIX2_h; }

private:
  uint32_t nClusters_h = 0;
  int32_t offsetBPIX2_h = 0;
};

namespace ngt {

  template <>
  struct MemoryCopyTraits<SiPixelClustersHost> {
    using value_type = SiPixelClustersHost;

    struct Properties {
      int size;
      uint32_t nClusters;
      int32_t offsetBPIX2;
    };

    static Properties properties(value_type const& object) {
      return {object->metadata().size() - 1, object.nClusters(), object.offsetBPIX2()};
    }

    static void initialize(value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pageable host memory.
      object = value_type(cms::alpakatools::host(), prop.size);
      object.setNClusters(prop.nClusters, prop.offsetBPIX2);
    }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pinned host memory.
      object = value_type(queue, prop.size);
      object.setNClusters(prop.nClusters, prop.offsetBPIX2);
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

#endif  // DataFormats_SiPixelClusterSoA_interface_SiPixelClustersHost_h
