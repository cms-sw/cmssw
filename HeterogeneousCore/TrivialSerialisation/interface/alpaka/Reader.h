#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_Reader_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_Reader_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TrivialSerialisation/interface/MemoryCopyTraits.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Common.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  // Reader for device products.
  template <typename T>
  class Reader : public ::ngt::ReaderBase {
    static_assert(::ngt::HasMemoryCopyTraits<T>, "No specialization of MemoryCopyTraits found for type T");

  public:
    using WrapperType = edm::Wrapper<T>;

    // Constructor from Wrapper<T>
    explicit Reader(WrapperType const& wrapper) : ::ngt::ReaderBase(&wrapper), productPtr_(&wrapper.bareProduct()) {}

    // Constructor from T const& directly.
    explicit Reader(T const& product) : ::ngt::ReaderBase(nullptr), productPtr_(&product) {}

    ::ngt::AnyBuffer parameters() const override { return ::ngt::get_properties<T>(*productPtr_); }

    std::vector<std::span<const std::byte>> regions() const override { return ::ngt::get_regions<T>(*productPtr_); }

  private:
    T const* productPtr_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_Reader_h
