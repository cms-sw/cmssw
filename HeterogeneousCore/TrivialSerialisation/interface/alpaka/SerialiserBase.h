#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserBase_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserBase_h

#include <memory>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  // Abstract interface for creating Readers and Writers for device products.
  class SerialiserBase {
  public:
    virtual std::unique_ptr<WriterBase> writer() = 0;
    virtual std::unique_ptr<const ReaderBase> reader(const edm::WrapperBase& wrapper) = 0;

    virtual ~SerialiserBase() = default;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserBase_h
