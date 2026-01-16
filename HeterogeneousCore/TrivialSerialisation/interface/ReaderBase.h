#ifndef TrivialSerialisation_Common_interface_ReaderBase_h
#define TrivialSerialisation_Common_interface_ReaderBase_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"

namespace ngt {

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

#endif  // TrivialSerialisation_Common_interface_ReaderBase_h
