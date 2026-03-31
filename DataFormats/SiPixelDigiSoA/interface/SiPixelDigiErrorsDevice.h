#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableDeviceCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

template <typename TDev>
class SiPixelDigiErrorsDevice : public PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA> {
public:
  SiPixelDigiErrorsDevice(edm::Uninitialized)
      : PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA>{edm::kUninitialized} {}

  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  explicit SiPixelDigiErrorsDevice(TQueue queue, size_t maxFedWords)
      : PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA>(queue, maxFedWords), maxFedWords_(maxFedWords) {}

  explicit SiPixelDigiErrorsDevice(TDev const& device, size_t maxFedWords)
      : PortableDeviceCollection<TDev, SiPixelDigiErrorsSoA>(device, maxFedWords) {}

  auto maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_;
};

// Specialize the MemoryCopyTraits for SiPixelDigiErrorsDevice
namespace ngt {

  template <typename TDev>
  struct MemoryCopyTraits<SiPixelDigiErrorsDevice<TDev>> {
    using value_type = SiPixelDigiErrorsDevice<TDev>;
    struct Properties {
      int maxFedWords;
    };

    static Properties properties(value_type const& object) { return {object.maxFedWords()}; }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& props) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in device global memory.
      object = value_type(queue, props.maxFedWords);
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

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsDevice_h
