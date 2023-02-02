#ifndef HeterogeneousCore_AlpakaCore_interface_ContextState_h
#define HeterogeneousCore_AlpakaCore_interface_ContextState_h

#include <memory>
#include <stdexcept>
#include <utility>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "HeterogeneousCore/AlpakaInterface/interface/ScopedContextFwd.h"
#include "HeterogeneousCore/AlpakaInterface/interface/traits.h"

namespace cms::alpakatools {

  /**
   * The purpose of this class is to deliver the device and queue
   * information from ExternalWork's acquire() to producer() via a
   * member/QueueCache variable.
   */
  template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  class ContextState {
  public:
    using Queue = TQueue;
    using Device = alpaka::Dev<Queue>;

    ContextState() = default;
    ~ContextState() = default;

    ContextState(const ContextState&) = delete;
    ContextState& operator=(const ContextState&) = delete;
    ContextState(ContextState&&) = delete;
    ContextState& operator=(ContextState&& other) = delete;

  private:
    friend class ScopedContextAcquire<TQueue>;
    friend class ScopedContextProduce<TQueue>;
    friend class ScopedContextTask<TQueue>;

    void set(std::shared_ptr<Queue> queue) {
      throwIfQueue();
      queue_ = std::move(queue);
    }

    Device device() const {
      throwIfNoQueue();
      return alpaka::getDev(*queue_);
    }

    Queue queue() const {
      throwIfNoQueue();
      return *queue_;
    }

    std::shared_ptr<Queue> const& queuePtr() const {
      throwIfNoQueue();
      return queue_;
    }

    std::shared_ptr<Queue> releaseQueuePtr() {
      throwIfNoQueue();
      // This function needs to effectively reset queue_ (i.e. queue_
      // must be empty after this function). This behavior ensures that
      // the std::shared_ptr<Queue> is not hold for inadvertedly long (i.e. to
      // the next event), and is checked at run time.
      Queue queue = std::move(queue_);
      return queue;
    }

    void throwIfQueue() const {
      if (queue_) {
        throw cms::Exception("LogicError") << "Trying to set ContextState, but it already had a valid state";
      }
    }

    void throwIfNoQueue() const {
      if (not queue_) {
        throw cms::Exception("LogicError") << "Trying to get ContextState, but it did not have a valid state";
      }
    }

    std::shared_ptr<Queue> queue_;
  };

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaCore_interface_ContextState_h
