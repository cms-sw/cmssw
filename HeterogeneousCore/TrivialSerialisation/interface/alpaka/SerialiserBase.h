#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserBase_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserBase_h

#include <any>
#include <functional>
#include <memory>
#include <typeinfo>

#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/WriterBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  // Abstract interface for creating Readers and Writers for device products.
  class SerialiserBase {
  public:
    virtual std::unique_ptr<WriterBase> writer() = 0;
    virtual std::unique_ptr<const ReaderBase> reader(const edm::WrapperBase& wrapper, EDMetadata& metadata) = 0;

    virtual bool hasCopyToHost() const = 0;
    virtual bool hasCopyToDevice() const = 0;

    // Methods needed to register a DtoH transform
    virtual std::function<std::any(Queue&, EDMetadata&, edm::WrapperBase const&)> preTransformDtoH() const = 0;
    virtual std::function<std::unique_ptr<edm::WrapperBase>(std::any const&)> transformDtoH() const = 0;
    virtual std::function<std::shared_ptr<Queue>(edm::WrapperBase const&)> getQueue() const = 0;

    // Methods needed to register a HtoD transform
    virtual std::function<std::any(Queue&, edm::WrapperBase const&)> preTransformHtoD() const = 0;
    virtual std::function<std::unique_ptr<edm::WrapperBase>(std::any const&, std::shared_ptr<EDMetadata>)>
    transformHtoD() const = 0;

    // Return the type_info of the product type (DeviceProduct<T> for async
    // backends, T for serial_sync)
    virtual std::type_info const& productTypeID() const = 0;

    // Return the type_info of the host-equivalent of T
    virtual std::type_info const& hostProductTypeID() const = 0;

    virtual ~SerialiserBase() = default;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_SerialiserBase_h
