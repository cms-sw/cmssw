#ifndef HeterogeneousCore_AlpakaInterface_interface_atomicIncSaturate_h
#define HeterogeneousCore_AlpakaInterface_interface_atomicIncSaturate_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

namespace cms::alpakatools {

  // Atomically increment the value at address by 1, unless it has already reached limit.
  // This is similar to alpaka::atomicInc, but instead of wrapping around to 0 it saturates at limit.
  // Return the old value: if it is greater than or equal to limit, the value at address was not changed.
  ALPAKA_NO_HOST_ACC_WARNING
  template <alpaka::concepts::Acc TAcc, typename T, typename THierarchy = alpaka::hierarchy::Grids>
    requires std::is_integral_v<T>
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T atomicIncSaturate(TAcc const& acc,
                                                          T* address,
                                                          std::type_identity_t<T> limit,
                                                          THierarchy const& hierarchy = THierarchy()) {
    // Read the current value atomically: a plain read would be a data race with the other threads.
    T old = alpaka::atomicAdd(acc, address, T{0}, hierarchy);
    T assumed;

    do {
      assumed = old;
      if (assumed >= limit) {
        // Saturate at limit.
        break;
      }
      old = alpaka::atomicCas(acc, address, assumed, static_cast<T>(assumed + 1), hierarchy);
    } while (old != assumed);

    return old;
  }

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_atomicIncSaturate_h
