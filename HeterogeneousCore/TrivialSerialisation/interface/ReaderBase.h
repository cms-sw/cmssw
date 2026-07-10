#ifndef HeterogeneousCore_TrivialSerialisation_interface_ReaderBase_h
#define HeterogeneousCore_TrivialSerialisation_interface_ReaderBase_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"

namespace ngt {

  // Base class for reading the memory regions and properties of a serialised product.
  class ReaderBase {
  public:
    ReaderBase(const edm::WrapperBase* ptr) : ptr_(ptr) {}

    virtual ngt::AnyBuffer parameters() const = 0;
    virtual std::vector<std::span<const std::byte>> regions() const = 0;

    virtual ~ReaderBase() = default;

  protected:
    const edm::WrapperBase* ptr_;
  };

}  // namespace ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_ReaderBase_h
