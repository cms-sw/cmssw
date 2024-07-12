#ifndef PhysicsTools_PyTorch_interface_config_h
#define PhysicsTools_PyTorch_interface_config_h

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

// Automatic translation of alpaka platform to torch constants. Embryon of PhysicsTools/PyTorch/interface/config.h
// We rely on HeterogeneousCore/AlpakaInterface/interface/config.h for filtering the defines and assume one and only
// one macro is defined among:
// ALPAKA_ACC_GPU_CUDA_ENABLED
// ALPAKA_ACC_GPU_HIP_ENABLED
// ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
// ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

namespace torch_common {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::CUDA;
#elif ALPAKA_ACC_GPU_HIP_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  constexpr c10::DeviceType kDeviceType = c10::DeviceType::CPU;
#else
  #error "Could not define the torch device type."
#endif
  // Torch stream setter for GPU backends (it's thread local in torch)
  // Also piggy backing on this to set the threading on CPUs
  template<typename TQueue>   
  class DeviceStreamGuard {
  public:
    DeviceStreamGuard(const TQueue &) {}
    void end() {}
    ~DeviceStreamGuard() {}
  };
  
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  // CUDA version
  template<>   
  class DeviceStreamGuard<alpaka_cuda_async::Queue> {
  public:
    using TQueue = alpaka_cuda_async::Queue;
    DeviceStreamGuard(const TQueue &queue): previous_(c10::cuda::getCurrentCUDAStream()) {
      torch::Device torchDevice(c10::DeviceType::CUDA, alpaka::getDev(queue).getNativeHandle());
      c10::cuda::CUDAStream torchStream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), torchDevice.index());
      c10::cuda::setCurrentCUDAStream(torchStream);
    }
    void end() {
      if (set_) {
        c10::cuda::setCurrentCUDAStream(previous_);
        set_ = false;
      }
    }
  private:
    c10::cuda::CUDAStream previous_;
    bool set_ = true;
  };
#endif
  
  // CPU versions
  template<typename TQueue>   
  class DeviceStreamGuardTorchThreadingDisabler {
  public:
    DeviceStreamGuardTorchThreadingDisabler(const TQueue &queue) {
      static TorchThreadingDisabler disabler;
    }
    void end() {}
    ~DeviceStreamGuardTorchThreadingDisabler() {}
  private:
    class TorchThreadingDisabler {
      TorchThreadingDisabler() {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
      }
    };
  };

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  template<>
  class DeviceStreamGuard<alpaka_serial_sync::Queue>: 
    public DeviceStreamGuardTorchThreadingDisabler <alpaka_serial_sync::Queue> {
    using DeviceStreamGuardTorchThreadingDisabler<alpaka_serial_sync::Queue>::DeviceStreamGuardTorchThreadingDisabler;
  };  
#endif
  
#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  template<>
  class DeviceStreamGuard<alpaka_tbb_async::Queue>: 
    public DeviceStreamGuardTorchThreadingDisabler <alpaka_tbb_async::Queue> {
    using DeviceStreamGuardTorchThreadingDisabler<alpaka_tbb_async::Queue>::DeviceStreamGuardTorchThreadingDisabler;
  };
#endif
  
  template<typename TBuf>
  torch::Tensor toTensor (TBuf &alpakaBuff) {
    // TODO: type is still hardcoded
    static_assert(1 == alpaka::Dim<TBuf>::value, "Current support limited to 1-dimension buffers");
    auto options = torch::TensorOptions().dtype(torch::kInt)
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      .device(c10::DeviceType::CUDA, alpaka::getDev(alpakaBuff).getNativeHandle())
#endif
      .pinned_memory(true);
    std::cout << "data type=" << typeid(*alpakaBuff.data()).name() << std::endl;
    std::cout << "data sizeof element=" << (unsigned)sizeof(*alpakaBuff.data()) << std::endl;
    std::cout << "buff extent product=" << alpaka::getExtentProduct(alpakaBuff) << std::endl;
    std::cout << "getExtends(buff)=["; 
    for (auto s : alpaka::getExtents(alpakaBuff)) {
      std::cout << s << ", ";
    }
    std::cout << "]" << std::endl;
    return torch::from_blob(alpakaBuff.data(), 
      {alpaka::getExtents(alpakaBuff)[0]},
      options);
  }
}

#endif // defined PhysicsTools_PyTorch_interface_config_h