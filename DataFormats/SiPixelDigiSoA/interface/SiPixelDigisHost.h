#ifndef DataFormats_SiPixelDigiSoA_interface_SiPixelDigisHost_h
#define DataFormats_SiPixelDigiSoA_interface_SiPixelDigisHost_h

#include "DataFormats/Common/interface/Uninitialized.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"

// TODO: The class is created via inheritance of the PortableDeviceCollection.
// This is generally discouraged, and should be done via composition.
// See: https://github.com/cms-sw/cmssw/pull/40465#discussion_r1067364306
class SiPixelDigisHost : public PortableHostCollection<SiPixelDigisSoA> {
public:
  explicit SiPixelDigisHost(edm::Uninitialized) : PortableHostCollection<SiPixelDigisSoA>{edm::kUninitialized} {}

  // Constructor for code that does not use alpaka explicitly, using the global
  // "host" object returned by cms::alpakatools::host().
  // Construct the object in pageable host memory.
  explicit SiPixelDigisHost(size_t maxFedWords)
      : PortableHostCollection<SiPixelDigisSoA>(cms::alpakatools::host(), maxFedWords + 1) {}

  // Construct the object in pageable host memory.
  explicit SiPixelDigisHost(alpaka_common::DevHost const& host, size_t maxFedWords)
      : PortableHostCollection<SiPixelDigisSoA>(host, maxFedWords + 1) {}

  // Construct the object in pinned host memory associated to the given work
  // queue, accessible by the queue's device.
  template <typename TQueue>
    requires(alpaka::isQueue<TQueue>)
  explicit SiPixelDigisHost(TQueue queue, size_t maxFedWords)
      : PortableHostCollection<SiPixelDigisSoA>(queue, maxFedWords + 1) {}

  void setNModules(uint32_t nModules) { nModules_h = nModules; }

  uint32_t nModules() const { return nModules_h; }
  uint32_t nDigis() const { return view().metadata().size() - 1; }

private:
  uint32_t nModules_h = 0;
};

// Specialize the MemoryCopyTraits for SiPixelDigisHost
namespace ngt {

  template <>
  struct MemoryCopyTraits<SiPixelDigisHost> {
    using value_type = SiPixelDigisHost;
    struct Properties {
      uint32_t nDigis;
      uint32_t nModules;
    };

    static Properties properties(value_type const& object) { return {object.nDigis(), object.nModules()}; }

    static void initialize(value_type& object, Properties const& props) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pageable host memory.
      object = value_type(cms::alpakatools::host(), props.nDigis);
      object.setNModules(props.nModules);
    }

    template <typename TQueue>
      requires(alpaka::isQueue<TQueue>)
    static void initialize(TQueue& queue, value_type& object, Properties const& props) {
      // Replace the default-constructed empty object with one where the buffer
      // has been allocated in pinned host memory.
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

#endif  // DataFormats_SiPixelDigiSoA_interface_SiPixelDigisHost_h
