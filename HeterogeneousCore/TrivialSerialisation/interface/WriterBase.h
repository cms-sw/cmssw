#ifndef HeterogeneousCore_TrivialSerialisation_interface_WriterBase_h
#define HeterogeneousCore_TrivialSerialisation_interface_WriterBase_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"

namespace ngt {

  // Base class for creating a new product and exposing its memory regions for writing.
  class WriterBase {
  public:
    WriterBase() = default;
    virtual ~WriterBase() = default;

    virtual void initialize(ngt::AnyBuffer const& args) = 0;
    virtual ngt::AnyBuffer uninitialized_parameters() const = 0;
    virtual std::vector<std::span<std::byte>> regions() = 0;
    virtual void finalize() = 0;

    std::unique_ptr<edm::WrapperBase> get() { return std::move(ptr_); }

  protected:
    std::unique_ptr<edm::WrapperBase> ptr_;
  };

}  // namespace ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_WriterBase_h
