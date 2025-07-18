#ifndef HeterogeneousCore_AlpakaInterface_interface_moveToDeviceAsync_h
#define HeterogeneousCore_AlpakaInterface_interface_moveToDeviceAsync_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

namespace cms::alpakatools {
  /**
   * This function moves the argument hostObject object to the device
   * specified by the queue. Here the "move" means that the argument
   * host object must not be used in the caller after this function
   * has been called.
   *
   * The CopyToDevice class template is used to define the returned
   * device object that corresponds the argument host object. For host
   * device the copying is skipped, and the hostData is returned directly.
   *
   * The host object must either
   * - allocate its memory in queue-ordered way (e.g. using make_host_buffer(TQueue, ...)), or
   * - synchronize in its destructor (makes this function synchronous, so not preferred)
   * If the host object uses non-queue-order-allocated memory, and
   * does not synchronize in its destructor, behavior is undefined.
   *
   * Note that the host object type is required to be non-copyable.
   * This is to avoid easy mistakes with objects that follow copy
   * semantics of std::shared_ptr (that includes Alpaka buffers), that
   * would allow the source memory buffer to be used via another copy
   * during the asynchronous data copy to the device.
   */
  template <typename TQueue, typename THostObject, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
  auto moveToDeviceAsync(TQueue& queue, THostObject hostObject) {
    static_assert(not(std::is_copy_constructible_v<THostObject> or std::is_copy_assignable_v<THostObject>),
                  "The data object to be moved to device must not be copyable.");

    if constexpr (std::is_same_v<alpaka::Dev<TQueue>, alpaka_common::DevHost>) {
      return hostObject;
    } else {
      return CopyToDevice<THostObject>::copyAsync(queue, hostObject);
    }
  }
}  // namespace cms::alpakatools

#endif
