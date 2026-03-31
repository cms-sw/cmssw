#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h

#include <memory>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "DataFormats/AlpakaCommon/interface/alpaka/DeviceProductType.h"
#include "DataFormats/AlpakaCommon/interface/alpaka/EDMetadata.h"
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
    using WrapperType = edm::Wrapper<detail::DeviceProductType<T>>;

    std::unique_ptr<WriterBase> writer() override { return std::make_unique<Writer<T>>(); }

    std::unique_ptr<const ReaderBase> reader(edm::WrapperBase const& wrapper,
                                             EDMetadata& metadata,
                                             bool tryReuseQueue) override {
      WrapperType const& w = dynamic_cast<WrapperType const&>(wrapper);
      if constexpr (detail::useProductDirectly) {
        return std::make_unique<Reader<T>>(w.bareProduct());
      } else {
        return std::make_unique<Reader<T>>(
            w.bareProduct().template getSynchronized<EDMetadata>(metadata, tryReuseQueue));
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
