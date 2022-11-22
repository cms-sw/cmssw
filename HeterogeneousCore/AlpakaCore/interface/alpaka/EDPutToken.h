#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EDPutToken_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EDPutToken_h

#include "DataFormats/Common/interface/DeviceProduct.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceProductType.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::device {
  class Event;

  /**
   * The device::EDPutToken is similar to edm::EDPutTokenT, but is
   * intended for Event data products in the device memory space
   * defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE). It
   * can be used only to put data into a device::Event
   */
  template <typename TProduct>
  class EDPutToken {
    using ProductType = typename detail::DeviceProductType<TProduct>::type;

  public:
    constexpr EDPutToken() noexcept = default;

    template <typename TAdapter>
    explicit EDPutToken(TAdapter&& adapter) : token_(adapter.template deviceProduces<TProduct, ProductType>()) {}

    template <typename TAdapter>
    EDPutToken& operator=(TAdapter&& adapter) {
      edm::EDPutTokenT<ProductType> tmp(adapter.template deviceProduces<TProduct, ProductType>());
      token_ = tmp;
      return *this;
    }

  private:
    friend class Event;

    auto const& underlyingToken() const { return token_; }

    edm::EDPutTokenT<ProductType> token_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::device

#endif
