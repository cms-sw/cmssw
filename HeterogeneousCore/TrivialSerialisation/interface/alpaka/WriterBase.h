#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_WriterBase_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_WriterBase_h

#include <cstddef>
#include <span>
#include <vector>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/AnyBuffer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  // Base class for creating a new device product and exposing its memory regions for writing
  class WriterBase {
  public:
    WriterBase() = default;
    virtual ~WriterBase() = default;

    virtual void initialize(Queue& queue, ::ngt::AnyBuffer const& args) = 0;
    virtual ::ngt::AnyBuffer uninitialized_parameters() const = 0;
    virtual std::vector<std::span<std::byte>> regions() = 0;
    virtual void finalize() = 0;

    virtual std::unique_ptr<edm::WrapperBase> get() = 0;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_WriterBase_h
