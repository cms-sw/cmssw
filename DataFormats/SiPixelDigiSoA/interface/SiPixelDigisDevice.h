#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class SiPixelDigisDevice : public PortableDeviceCollection<TDev, SiPixelDigisSoA> {
public:
  explicit SiPixelDigisDevice(edm::Uninitialized)
      : PortableDeviceCollection<TDev, SiPixelDigisSoA>{edm::kUninitialized} {}

  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  explicit SiPixelDigisDevice(TQueue queue, size_t maxFedWords)
      : PortableDeviceCollection<TDev, SiPixelDigisSoA>(queue, maxFedWords + 1) {}

  explicit SiPixelDigisDevice(TDev const& device, size_t maxFedWords)
      : PortableDeviceCollection<TDev, SiPixelDigisSoA>(device, maxFedWords + 1) {}

  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return this->view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;
};

// Specialize the MemoryCopyTraits for SiPixelDigisDevice
namespace ngt {

  template <typename TDev>
  struct MemoryCopyTraits<SiPixelDigisDevice<TDev>> {
    using value_type = SiPixelDigisDevice<TDev>;
    struct Properties {
      uint32_t nDigis;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nDigis(), object.nModules()}; }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& props) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in global device memory.
      object = value_type(queue, props.nDigis);
      object.setNModules(props.nModules);
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

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigisDevice_h
