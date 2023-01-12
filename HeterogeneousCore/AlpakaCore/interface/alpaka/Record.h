#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_Record_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_Record_h

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/QueueCache.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProduct.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESDeviceProductType.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class ESProducer;

  namespace device {
    /**
   * The device::Record class template mimics the EventSetup Record
   * classes, and provides access to ESProducts in the host memory
   * space, and in the device memory space defined by the backend
   * (i.e. ALPAKA_ACCELERATOR_NAMESPACE), that exist in the TRecord
   * Record. The device::Record also gives access to the Queue object
   * the ESProducer should use to queue all the device operations.
   *
   * Access to device memory space products is synchronized properly.
   *
   * Note that not full interface of EventSetup Record is replicated
   * here. If something important is missing, that can be added.
   */
    template <typename TRecord>
    class Record {
    public:
      // TODO: support for multiple devices will be added later
      Record(TRecord const& record, Device const& device)
          : record_(record), queue_(cms::alpakatools::getQueueCache<Queue>().get(device)) {}

      // Alpaka operations do not accept a temporary as an argument
      // TODO: Returning non-const reference here is BAD
      Queue& queue() const { return *queue_; }

      // getHandle()

      template <typename TProduct, typename TDepRecord>
      edm::ESHandle<TProduct> getHandle(edm::ESGetToken<TProduct, TDepRecord> const& iToken) const {
        return record_.getHandle(iToken);
      }

      template <typename TProduct, typename TDepRecord>
      edm::ESHandle<TProduct> getHandle(device::ESGetToken<TProduct, TDepRecord> const& iToken) const {
        auto handle = record_.getHandle(iToken.underlyingToken());
        if constexpr (detail::useESProductDirectly<TProduct>) {
          return handle;
        } else {
          if (not handle) {
            return edm::ESHandle<TProduct>(handle.whyFailedFactory());
          }
          return edm::ESHandle<TProduct>(&handle->get(alpaka::getDev(*queue_)), handle.description());
        }
      }

      // getTransientHandle()

      template <typename TProduct, typename TDepRecord>
      edm::ESTransientHandle<TProduct> getTransientHandle(edm::ESGetToken<TProduct, TDepRecord> const& iToken) const {
        return record_.getTransientHandle(iToken);
      }

      template <typename TProduct, typename TDepRecord>
      edm::ESTransientHandle<TProduct> getTransientHandle(device::ESGetToken<TProduct, TDepRecord> const& iToken) const {
        auto handle = record_.getTransientHandle(iToken.underlyingToken());
        if constexpr (detail::useESProductDirectly<TProduct>) {
          return handle;
        } else {
          if (not handle) {
            return edm::ESTransientHandle<TProduct>();
          }
          if (handle.failedToGet()) {
            return edm::ESTransientHandle<TProduct>(handle.whyFailedFactory());
          }
          return edm::ESTransientHandle<TProduct>(&handle->get(alpaka::getDev(*queue_)), handle.description());
        }
      }

      // get()

      template <typename TProduct, typename TDepRecord>
      TProduct const& get(edm::ESGetToken<TProduct, TDepRecord> const& iToken) const {
        return record_.get(iToken);
      }

      template <typename TProduct, typename TDepRecord>
      TProduct const& get(device::ESGetToken<TProduct, TDepRecord> const& iToken) const {
        auto const& product = record_.get(iToken.underlyingToken());
        if constexpr (detail::useESProductDirectly<TProduct>) {
          return product;
        } else {
          return product.get(alpaka::getDev(*queue_));
        }
      }

    private:
      friend ESProducer;

      std::shared_ptr<Queue> queuePtr() const { return queue_; }

      TRecord const& record_;
      std::shared_ptr<Queue> queue_;
    };
  }  // namespace device
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
