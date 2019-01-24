#ifndef CUDADataFormats_Common_interface_device_unique_ptr_h
#define CUDADataFormats_Common_interface_device_unique_ptr_h

#include <memory>
#include <functional>

namespace edm {
  namespace cuda {
    namespace device {
      template <typename T>
      using unique_ptr = std::unique_ptr<T, std::function<void(void *)>>;
    }
  }
}

#endif
