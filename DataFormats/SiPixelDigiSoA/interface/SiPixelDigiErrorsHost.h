#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h

#include <utility>

#include <alpaka/alpaka.hpp>

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelRawData/interface/SiPixelErrorCompact.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/SimpleVector.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

class SiPixelDigiErrorsHost : public PortableHostCollection<SiPixelDigiErrorsSoA> {
public:
  explicit SiPixelDigiErrorsHost(edm::Uninitialized)
      : PortableHostCollection<SiPixelDigiErrorsSoA>{edm::kUninitialized} {}

  // Constructor for code that does not use alpaka explicitly, using the global
  // "host" object returned by cms::alpakatools::host().
  // Construct the object in pageable host memory.
  explicit SiPixelDigiErrorsHost(size_t maxFedWords)
      : PortableHostCollection<SiPixelDigiErrorsSoA>(cms::alpakatools::host(), maxFedWords),
        maxFedWords_(maxFedWords) {}

  // Construct the object in pageable host memory.
  explicit SiPixelDigiErrorsHost(alpaka_common::DevHost const& host, size_t maxFedWords)
      : PortableHostCollection<SiPixelDigiErrorsSoA>(host, maxFedWords), maxFedWords_(maxFedWords) {}

  // Construct the object in pinned host memory associated to the given work
  // queue, accessible by the queue's device.
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  explicit SiPixelDigiErrorsHost(TQueue queue, int maxFedWords)
      : PortableHostCollection<SiPixelDigiErrorsSoA>(queue, maxFedWords), maxFedWords_(maxFedWords) {}

  int maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_ = 0;
};

// Specialize the MemoryCopyTraits for SiPixelDigiErrorsHost
namespace ngt {

  template <>
  struct MemoryCopyTraits<SiPixelDigiErrorsHost> {
    using value_type = SiPixelDigiErrorsHost;
    struct Properties {
      int maxFedWords;
    };

    static Properties properties(value_type const& object) { return {object.maxFedWords()}; }

    static void initialize(value_type& object, Properties const& props) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pageable host memory.
      object = value_type(cms::alpakatools::host(), props.maxFedWords);
    }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& props) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pinned host memory.
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

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigiErrorsHost_h
