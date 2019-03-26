#ifndef HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_host_unique_ptr_h

#include <memory>
#include <functional>

namespace cudautils {
  namespace host {
    namespace impl {
      // Additional layer of types to distinguish from host::unique_ptr
      class HostDeleter {
      public:
        HostDeleter() = default;
        explicit HostDeleter(std::function<void(void *)> f): f_(f) {}

        void operator()(void *ptr) { f_(ptr); }
      private:
        std::function<void(void *)> f_;
      };
    }

    template <typename T>
    using unique_ptr = std::unique_ptr<T, impl::HostDeleter>;
  }
}

#endif
