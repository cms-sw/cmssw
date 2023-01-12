#ifndef HeterogeneousCore_AlpakaCore_interface_alpaka_Event_h
#define HeterogeneousCore_AlpakaCore_interface_alpaka_Event_h

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/DeviceProductType.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDMetadata.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::device {
  /**
   * The device::Event mimics edm::Event, and provides access to
   * EDProducts in the host memory space, and in the device memory
   * space defined by the backend (i.e. ALPAKA_ACCELERATOR_NAMESPACE).
   * The device::Event also gives access to the Queue object the
   * EDModule code should use to queue all the device operations.
   *
   * Access to device memory space products is synchronized properly.
   * For backends with synchronous Queue this is trivial. For
   * asynchronous Queue, either the Queue of the EDModule is taken
   * from the first data product, or a wait is inserted into the
   * EDModule's Queue to wait for the product's asynchronous
   * production to finish.
   *
   * Note that not full interface of edm::Event is replicated here. If
   * something important is missing, that can be added.
   */
  class Event {
  public:
    // To be called in produce()
    explicit Event(edm::Event& ev, std::shared_ptr<EDMetadata> metadata)
        : constEvent_(ev), event_(&ev), metadata_(std::move(metadata)) {}

    // To be called in acquire()
    explicit Event(edm::Event const& ev, std::shared_ptr<EDMetadata> metadata)
        : constEvent_(ev), metadata_(std::move(metadata)) {}

    Event(Event const&) = delete;
    Event& operator=(Event const&) = delete;
    Event(Event&&) = delete;
    Event& operator=(Event&&) = delete;

    auto streamID() const { return constEvent_.streamID(); }
    auto id() const { return constEvent_.id(); }

    Device device() const { return metadata_->device(); }

    // Alpaka operations do not accept a temporary as an argument
    // TODO: Returning non-const reference here is BAD
    Queue& queue() const {
      queueUsed_ = true;
      return metadata_->queue();
    }

    // get()

    template <typename T>
    T const& get(edm::EDGetTokenT<T> const& token) const {
      return constEvent_.get(token);
    }

    template <typename T>
    T const& get(device::EDGetToken<T> const& token) const {
      auto const& deviceProduct = constEvent_.get(token.underlyingToken());
      if constexpr (detail::useProductDirectly<T>) {
        return deviceProduct;
      } else {
        // try to re-use queue from deviceProduct if our queue has not yet been used
        T const& product = deviceProduct.template getSynchronized<EDMetadata>(*metadata_, not queueUsed_);
        queueUsed_ = true;
        return product;
      }
    }

    // getHandle()

    template <typename T>
    edm::Handle<T> getHandle(edm::EDGetTokenT<T> const& token) const {
      return constEvent_.getHandle(token);
    }

    template <typename T>
    edm::Handle<T> getHandle(device::EDGetToken<T> const& token) const {
      auto deviceProductHandle = constEvent_.getHandle(token.underlyingToken());
      if constexpr (detail::useProductDirectly<T>) {
        return deviceProductHandle;
      } else {
        if (not deviceProductHandle) {
          return edm::Handle<T>(deviceProductHandle.whyFailedFactory());
        }
        // try to re-use queue from deviceProduct if our queue has not yet been used
        T const& product = deviceProductHandle->getSynchronized(*metadata_, not queueUsed_);
        queueUsed_ = true;
        return edm::Handle<T>(&product, deviceProductHandle.provenance());
      }
    }

    // emplace()

    template <typename T, typename... Args>
    edm::OrphanHandle<T> emplace(edm::EDPutTokenT<T> const& token, Args&&... args) {
      return event_->emplace(token, std::forward<Args>(args)...);
    }

    // TODO: what to do about the returned OrphanHandle object?
    // The idea for Ref-like things in this domain differs from earlier Refs anyway
    template <typename T, typename... Args>
    void emplace(device::EDPutToken<T> const& token, Args&&... args) {
      if constexpr (detail::useProductDirectly<T>) {
        event_->emplace(token.underlyingToken(), std::forward<Args>(args)...);
      } else {
        event_->emplace(token.underlyingToken(), metadata_, std::forward<Args>(args)...);
      }
    }

    // put()

    template <typename T>
    edm::OrphanHandle<T> put(edm::EDPutTokenT<T> const& token, std::unique_ptr<T> product) {
      return event_->put(token, std::move(product));
    }

    template <typename T>
    void put(device::EDPutToken<T> const& token, std::unique_ptr<T> product) {
      if constexpr (detail::useProductDirectly<T>) {
        event_->emplace(token.underlyingToken(), std::move(*product));
      } else {
        event_->emplace(token.underlyingToken(), metadata_, std::move(*product));
      }
    }

  private:
    // Having both const and non-const here in order to serve the
    // clients with one device::Event class
    edm::Event const& constEvent_;
    edm::Event* event_ = nullptr;

    std::shared_ptr<EDMetadata> metadata_;
    // device::Event is not supposed to be const-thread-safe, so no
    // additional protection is needed.
    mutable bool queueUsed_ = false;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::device

#endif
