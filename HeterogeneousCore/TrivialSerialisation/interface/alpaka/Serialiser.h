#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h

#include <memory>
#include <type_traits>

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Reader.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/alpaka/Writer.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt {

  // Concrete Serialiser for device products.
  // T is the inner product type (e.g. PortableDeviceCollection<...>).
  template <typename T>
  class Serialiser : public SerialiserBase {
  public:
    using DeviceProductType = edm::DeviceProduct<T>;
    using WrapperType = edm::Wrapper<DeviceProductType>;

    std::unique_ptr<WriterBase> writer() override { return std::make_unique<Writer<T>>(); }

    std::unique_ptr<const ReaderBase> reader(edm::WrapperBase const& wrapper) override {
      // The wrapper contains DeviceProduct<T>; extract the inner T const&
      WrapperType const& w = dynamic_cast<WrapperType const&>(wrapper);
      return std::make_unique<Reader<T>>(w.bareProduct().getSynchronized());
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
