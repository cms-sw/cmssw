#ifndef DataFormats_Portable_interface_ProductBase_h
#define DataFormats_Portable_interface_ProductBase_h

#include <atomic>
#include <memory>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/ScopedContextFwd.h"

namespace cms::alpakatools {

  /**
   * Base class for all instantiations of Product<TQueue, T> to hold the
   * non-T-dependent members.
   */
  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  class ProductBase {
  public:
    using Queue = TQueue;
    using Event = alpaka::Event<Queue>;
    using Device = alpaka::Dev<Queue>;

    ProductBase() = default;  // Needed only for ROOT dictionary generation

    ~ProductBase() {
      // Make sure that the production of the product in the GPU is
      // complete before destructing the product. This is to make sure
      // that the EDM stream does not move to the next event before all
      // asynchronous processing of the current is complete.

      // TODO: a callback notifying a WaitingTaskHolder (or similar)
      // would avoid blocking the CPU, but would also require more work.

      // FIXME: this may throw an execption if the underlaying call fails.
      if (event_) {
        alpaka::wait(*event_);
      }
    }

    ProductBase(const ProductBase&) = delete;
    ProductBase& operator=(const ProductBase&) = delete;
    ProductBase(ProductBase&& other)
        : queue_{std::move(other.queue_)}, event_{std::move(other.event_)}, mayReuseQueue_{other.mayReuseQueue_.load()} {}
    ProductBase& operator=(ProductBase&& other) {
      queue_ = std::move(other.queue_);
      event_ = std::move(other.event_);
      mayReuseQueue_ = other.mayReuseQueue_.load();
      return *this;
    }

    bool isValid() const { return queue_.get() != nullptr; }

    bool isAvailable() const {
      // if default-constructed, the product is not available
      if (not event_) {
        return false;
      }
      return alpaka::isComplete(*event_);
    }

    // returning a const& requires changes in alpaka's getDev() implementations
    Device device() const { return alpaka::getDev(queue()); }

    Queue const& queue() const { return *queue_; }

    Event const& event() const { return *event_; }

  protected:
    explicit ProductBase(std::shared_ptr<Queue> queue, std::shared_ptr<Event> event)
        : queue_{std::move(queue)}, event_{std::move(event)} {}

  private:
    friend class impl::ScopedContextBase<Queue>;
    friend class ScopedContextProduce<Queue>;

    // The following function is intended to be used only from ScopedContext
    const std::shared_ptr<Queue>& queuePtr() const { return queue_; }

    bool mayReuseQueue() const {
      bool expected = true;
      bool changed = mayReuseQueue_.compare_exchange_strong(expected, false);
      // If the current thread is the one flipping the flag, it may
      // reuse the queue.
      return changed;
    }

    // shared_ptr because of caching in QueueCache, and sharing across edm::Event products
    std::shared_ptr<Queue> queue_;  //!
    // shared_ptr because of caching in EventCache
    std::shared_ptr<Event> event_;  //!

    // This flag tells whether the queue may be reused by a consumer or not.
    // The goal is to have a "chain" of modules to enqueue their work to the same queue.
    mutable std::atomic<bool> mayReuseQueue_ = true;  //!
  };

}  // namespace cms::alpakatools

#endif  // DataFormats_Portable_interface_ProductBase_h
