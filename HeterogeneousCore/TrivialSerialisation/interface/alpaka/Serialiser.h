#ifndef HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
#define HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h

#include <memory>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceProductType.h"
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

    std::unique_ptr<const ReaderBase> reader(edm::WrapperBase const& wrapper) override {
      WrapperType const& w = dynamic_cast<WrapperType const&>(wrapper);
      if constexpr (detail::useProductDirectly) {
        return std::make_unique<Reader<T>>(w.bareProduct());
      } else {
        // On Device queues, w.bareProduct() returns edm::DeviceProduct<T>.
        // getUnsynchronized() then extracts T without performing any
        // synchronization. This is ok for the purpose of getting the memory
        // regions from T, but the caller must ensure synchronization before reading
        // data from them.
        return std::make_unique<Reader<T>>(w.bareProduct().getUnsynchronized());
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ngt

#endif  // HeterogeneousCore_TrivialSerialisation_interface_alpaka_Serialiser_h
