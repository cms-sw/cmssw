#ifndef HeterogeneousCore_AlpakaInterface_interface_ScopedContextFwd_h
#define HeterogeneousCore_AlpakaInterface_interface_ScopedContextFwd_h

#include <alpaka/alpaka.hpp>

// Forward declaration of the alpaka framework Context classes
//
// This file is under HeterogeneousCore/AlpakaInterface to avoid introducing a dependency on
// HeterogeneousCore/AlpakaCore.

namespace cms::alpakatools {

  namespace impl {
    template <typename TQueue, typename = std::enable_if_t<alpaka::concepts::Queue<TQueue>>>
    class ScopedContextBase;

    template <typename TQueue, typename = std::enable_if_t<alpaka::concepts::Queue<TQueue>>>
    class ScopedContextGetterBase;
  }  // namespace impl

  template <typename TQueue, typename = std::enable_if_t<alpaka::concepts::Queue<TQueue>>>
  class ScopedContextAcquire;

  template <typename TQueue, typename = std::enable_if_t<alpaka::concepts::Queue<TQueue>>>
  class ScopedContextProduce;

  template <typename TQueue, typename = std::enable_if_t<alpaka::concepts::Queue<TQueue>>>
  class ScopedContextTask;

  template <typename TQueue, typename = std::enable_if_t<alpaka::concepts::Queue<TQueue>>>
  class ScopedContextAnalyze;

}  // namespace cms::alpakatools

#endif  // HeterogeneousCore_AlpakaInterface_interface_ScopedContextFwd_h
