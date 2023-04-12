#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_EventSetup_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_EventSetup_h

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProductType.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::device {
  /**
   * The device::EventSetup mimics edm::EventSetup, and provides access
   * to ESProducts in the host memory space, and in the device memory
   * space defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE).
   *
   * Access to device memory space products is synchronized properly.
   *
   * Note that not full interface of edm::EventSetup is replicated
   * here. If something important is missing, that can be added.
   */
  class EventSetup {
  public:
    EventSetup(edm::EventSetup const& iSetup, Device const& dev) : setup_(iSetup), device_(dev) {}

    // To be able to interact with non-Alpaka helper code that needs
    // to access edm::EventSetup
    operator edm::EventSetup const &() const { return setup_; }

    // getData()

    template <typename T, typename R>
    T const& getData(edm::ESGetToken<T, R> const& iToken) const {
      return setup_.getData(iToken);
    }

    template <typename T, typename R>
    T const& getData(device::ESGetToken<T, R> const& iToken) const {
      auto const& product = setup_.getData(iToken.underlyingToken());
      if constexpr (detail::useESProductDirectly<T>) {
        return product;
      } else {
        return product.get(device_);
      }
    }

    // getHandle()

    template <typename T, typename R>
    edm::ESHandle<T> getHandle(edm::ESGetToken<T, R> const& iToken) const {
      return setup_.getHandle(iToken);
    }

    template <typename T, typename R>
    edm::ESHandle<T> getHandle(device::ESGetToken<T, R> const& iToken) const {
      auto handle = setup_.getHandle(iToken.underlyingToken());
      if constexpr (detail::useESProductDirectly<T>) {
        return handle;
      } else {
        if (not handle) {
          return edm::ESHandle<T>(handle.whyFailedFactory());
        }
        return edm::ESHandle<T>(&handle->get(device_), handle.description());
      }
    }

    // getTransientHandle() is intentionally omitted for now. It makes
    // little sense for event transitions, and for now
    // device::EventSetup is available only for those. If
    // device::EventSetup ever gets added for run or lumi transitions,
    // getTransientHandle() will be straightforward to add

  private:
    edm::EventSetup const& setup_;
    // Taking a copy because alpaka::getDev() returns a temporary. To
    // be removed after a proper treatment of multiple devices per
    // backend is implemented in Eventsetup
    Device const device_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::device

#endif
