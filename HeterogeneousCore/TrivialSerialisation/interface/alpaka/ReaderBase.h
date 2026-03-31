#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_ReaderBase_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_ReaderBase_h

#include <cstddef>
#include <span>
#include <vector>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  // Base class for reading the memory regions and properties of a serialised device product.
  class ReaderBase {
  public:
    ReaderBase() = default;

    virtual ::ngt::AnyBuffer parameters() const = 0;
    virtual std::vector<std::span<const std::byte>> regions() const = 0;

    virtual ~ReaderBase() = default;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_ReaderBase_h
