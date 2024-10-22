#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_ESGetToken_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_ESGetToken_h

#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProductType.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::device {
  class EventSetup;
  template <typename T>
  class Record;

  /**
   * The device::ESGetToken is similar to edm::ESGetToken, but is
   * intended for EventSetup data products in the device memory space
   * defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE). It
   * can be used only to get data from a device::EventSetup and
   * device::Record<T>.
   */
  template <typename ESProduct, typename ESRecord>
  class ESGetToken {
  public:
    constexpr ESGetToken() noexcept = default;

    template <typename TAdapter>
    constexpr ESGetToken(TAdapter&& iAdapter) : token_(std::forward<TAdapter>(iAdapter)) {}

  private:
    friend class EventSetup;
    template <typename T>
    friend class Record;

    auto const& underlyingToken() const { return token_; }

    using ProductType = typename detail::ESDeviceProductType<ESProduct>::type;
    edm::ESGetToken<ProductType, ESRecord> token_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::device

#endif
