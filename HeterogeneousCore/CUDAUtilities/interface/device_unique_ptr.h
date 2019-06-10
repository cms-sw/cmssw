#ifndef HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h
#define HeterogeneousCore_CUDAUtilities_interface_device_unique_ptr_h

#include <memory>
#include <functional>

namespace cudautils {
  namespace device {
    namespace impl {
      // Additional layer of types to distinguish from host::unique_ptr
      class DeviceDeleter {
      public:
        DeviceDeleter() = default;
        explicit DeviceDeleter(std::function<void(void *)> f): f_(f) {}

        void operator()(void *ptr) { f_(ptr); }
      private:
        std::function<void(void *)> f_;
      };
    }

    template <typename T>
    using unique_ptr = std::unique_ptr<T, impl::DeviceDeleter>;
  }
}

#endif
