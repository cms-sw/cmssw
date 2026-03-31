#ifndef DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h
#define DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersHost.h"
#include "DataFormats/SiPixelClusterSoA/interface/SiPixelClustersSoA.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToHost.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class SiPixelClustersDevice : public PortableDeviceCollection<TDev, SiPixelClustersSoA> {
public:
  SiPixelClustersDevice(edm::Uninitialized) : PortableDeviceCollection<TDev, SiPixelClustersSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit SiPixelClustersDevice(TQueue queue, size_t maxModules)
      : PortableDeviceCollection<TDev, SiPixelClustersSoA>(queue, maxModules + 1) {}

  // Constructor which specifies the SoA size
  explicit SiPixelClustersDevice(TDev const& device, size_t maxModules)
      : PortableDeviceCollection<TDev, SiPixelClustersSoA>(device, maxModules + 1) {}

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

  // Specialize MemoryCopyTraits for SiPixelClustersDevice
  template <typename TDev>
  struct MemoryCopyTraits<SiPixelClustersDevice<TDev>> {
    using value_type = SiPixelClustersDevice<TDev>;

    struct Properties {
      int size;
      uint32_t nClusters;
      int32_t offsetBPIX2;
    };

    static Properties properties(value_type const& object) {
      return {object->metadata().size() - 1, object.nClusters(), object.offsetBPIX2()};
    }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& prop) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in device global memory.
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

#endif  // DataFormats_SiPixelClusterSoA_interface_SiPixelClustersDevice_h
