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
  SiPixelDigiErrorsHost(edm::Uninitialized) : PortableHostCollection<SiPixelDigiErrorsSoA>{edm::kUninitialized} {}

  template <typename TQueue>
  explicit SiPixelDigiErrorsHost(int maxFedWords, TQueue queue)
      : PortableHostCollection<SiPixelDigiErrorsSoA>(maxFedWords, queue), maxFedWords_(maxFedWords) {}

  int maxFedWords() const { return maxFedWords_; }

private:
  int maxFedWords_ = 0;
};

// Specialize the MemoryCopyTraits for SiPixelDigisHost
namespace ngt {

  template <>
  struct MemoryCopyTraits<SiPixelDigiErrorsHost> {
    using value_type = SiPixelDigiErrorsHost;
    struct Properties {
      int maxFedWords;
    };

    static Properties properties(value_type const& object) { return {object.maxFedWords()}; }

    static void initialize(value_type& object, Properties const& props) {
      // replace the default-constructed empty object with one where the buffer has been allocated in pageable system memory
      object = value_type(props.maxFedWords, cms::alpakatools::host());
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
