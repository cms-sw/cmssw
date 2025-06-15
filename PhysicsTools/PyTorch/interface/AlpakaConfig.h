// ROCm/HIP backend not yet supported, see: https://github.com/pytorch/pytorch/blob/main/aten/CMakeLists.txt#L75

#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_

#include <alpaka/alpaka.hpp>
#include "PhysicsTools/PyTorch/interface/Config.h"

// #ifdef ALPAKA_ACC_GPU_CUDA_ENABLED || ALPAKA_ACC_GPU_HIP_ENABLED
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <c10/cuda/CUDAStream.h>
#endif

namespace cms::torch::alpaka {

  template <typename>
  inline constexpr bool false_value = false;

/**
 * @brief Specifies the device type used in the torch integration with Alpaka.
 * 
 * Depending on the available backend (CUDA or CPU), this defines the appropriate
 * PyTorch device type (`c10::DeviceType`) for the system.
 */
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
  constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CUDA;
// #elif ALPAKA_ACC_GPU_HIP_ENABLED
// constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::HIP;
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
  constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
  constexpr c10::DeviceType kTorchDeviceType = c10::DeviceType::CPU;
#else
#error "Could not define the torch device type."
#endif

  /**
 * @brief Converts an Alpaka device or queue object to a PyTorch device.
 * 
 * This function extracts the native handle from the Alpaka device or queue and
 * converts it into the corresponding PyTorch device.
 * 
 * TODO: query and map from device not Alpaka accelerator CMake flag
 * 
 * @tparam T The type of Alpaka object (Device or Queue).
 * @param obj The Alpaka object (Device or Queue) to convert.
 * @return Corresponding PyTorch device.
 */
  template <typename T>
  inline ::torch::Device device(const T &obj) {
    if constexpr (::alpaka::isDevice<T>)
      return ::torch::Device(kTorchDeviceType, obj.getNativeHandle());
    else if constexpr (::alpaka::isQueue<T>)
      return ::torch::Device(kTorchDeviceType, ::alpaka::getDev(obj).getNativeHandle());
    else
      static_assert(false_value<T>, "Unsupported type passed to device()");
  }

  /**
   * @brief Computes a unique hash representation for the given Alpaka queue.
   * 
   * This function generates a hash string that uniquely represents the given Alpaka queue.
   * For debugging purposes, to identify different queues in multi-threaded environments.
   * 
   * @tparam TQueue The type of Alpaka queue.
   * @param queue The Alpaka queue to generate the hash for.
   * @return A string representing the unique hash for the queue.
   */
  template <typename TQueue, typename = std::enable_if_t<::alpaka::isQueue<TQueue>>>
  inline std::string queue_hash(const TQueue &queue) {
    std::stringstream repr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    auto stream = c10::cuda::getStreamFromExternal(queue.getNativeHandle(), device(queue).index());
    repr << "0x" << std::hex << stream.stream();
    return repr.str();
// #elif ALPAKA_ACC_GPU_HIP_ENABLED
//   return "0x0";
#endif
    repr << "0x" << std::hex << std::hash<std::thread::id>{}(std::this_thread::get_id());
    return repr.str();
  }

  /**
   * @brief Computes a unique hash representation for the current stream associated with the given Alpaka queue.
   * 
   * This function generates a hash string representing the current stream of the given Alpaka queue,
   * For debugging purposes - identifying the currently active stream.
   * 
   * @tparam TQueue The type of Alpaka queue.
   * @param queue The Alpaka queue to generate the current stream hash for.
   * @return A string representing the unique hash for the current stream.
   */
  template <typename TQueue, typename = std::enable_if_t<::alpaka::isQueue<TQueue>>>
  inline std::string current_stream_hash(const TQueue &queue) {
    std::stringstream repr;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    const auto dev = device(queue);
    auto stream = c10::cuda::getCurrentCUDAStream(dev.index());
    repr << "0x" << std::hex << stream.stream();
    return repr.str();
// #elif ALPAKA_ACC_GPU_HIP_ENABLED
//     return "0x0";
#endif
    repr << "0x" << std::hex << std::hash<std::thread::id>{}(std::this_thread::get_id());
    return repr.str();
  }

  /**
   * @brief Sets the guard to disable multi-threading and control PyTorch's threading model.
   * @note Global call.
   * 
   * TODO: this should be called only once to disable PyTorch multi-threading, 
   *       consider moving to AlpakaService.
   */
  inline void set_threading_guard() {
    static std::once_flag threading_guard_flag;
    std::call_once(threading_guard_flag, [] {
      at::set_num_threads(1);
      at::set_num_interop_threads(1);
    });
  }

  /**
   * @brief Base class for managing guard scopes that ensure thread safety
   *        when working with Alpaka queues and PyTorch models.
   * 
   * Provides a mechanism to ensure correct PyTorch device and resources management.
   * @tparam TQueue The type of Alpaka queue.
   */
  template <typename T, typename TQueue>
  class GuardScope {
  public:
    explicit GuardScope(const TQueue &queue) : queue_(queue) { set(); }
    ~GuardScope() { reset(); };

  protected:
    const TQueue &queue_;

  private:
    /**
     * @brief Sets the guard to control PyTorch's execution model.
     */
    void set() { static_cast<T *>(this)->set_impl(); }

    /**
     * @brief Resets the guard state, restoring the previous configuration.
     */
    void reset() { static_cast<T *>(this)->reset_impl(); }
  };

  /**
   * @brief Specialization of GuardScope for different Alpaka queue types.
   */
  template <typename TQueue>
  struct GuardTraits;

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED

  /**
   * @brief Guard for CUDA-based operations when using Alpaka with the CUDA backend.
   * 
   * This class manages the stream switching required when running Alpaka queues on a CUDA device.
   * It ensures that the correct stream is set before operations and resets it afterward.
   */
  class CudaAsyncGuard : public GuardScope<CudaAsyncGuard, alpaka_cuda_async::Queue> {
  public:
    using Base = GuardScope<CudaAsyncGuard, alpaka_cuda_async::Queue>;
    using Base::Base;

    void set_impl() {
      prev_stream_ = c10::cuda::getCurrentCUDAStream();
      auto dev = device(this->queue_);
      auto stream = c10::cuda::getStreamFromExternal(this->queue_.getNativeHandle(), dev.index());
      c10::cuda::setCurrentCUDAStream(stream);
    }

    void reset_impl() { c10::cuda::setCurrentCUDAStream(prev_stream_); }

  private:
    c10::cuda::CUDAStream prev_stream_ = c10::cuda::getDefaultCUDAStream();
  };

  template <>
  struct GuardTraits<alpaka_cuda_async::Queue> {
    using type = CudaAsyncGuard;
  };
// #elif ALPAKA_ACC_GPU_HIP_ENABLED
// Similar structure can be added for HIP support, when CMSSW will support PyTorch build for ROCm.
// torch::cuda namespace is hip when using ROCm/HIP
#elif ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

  /**
   * @brief Guard for serial CPU operations using Alpaka with CPU serial backend.
   */
  class CpuSerialSyncGuard : public GuardScope<CpuSerialSyncGuard, alpaka_serial_sync::Queue> {
  public:
    using Base = GuardScope<CpuSerialSyncGuard, alpaka_serial_sync::Queue>;
    using Base::Base;
    void set_impl() { /**< nothing to be done */ }
    void reset_impl() { /**< nothing to be done */ }
  };

  template <>
  struct GuardTraits<alpaka_serial_sync::Queue> {
    using type = CpuSerialSyncGuard;
  };
#elif ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

  /**
   * @brief Guard for asynchronous CPU operations using Alpaka with TBB backend.
   */
  class CpuTbbAsyncGuard : public GuardScope<CpuTbbAsyncGuard, alpaka_tbb_async::Queue> {
  public:
    using Base = GuardScope<CpuTbbAsyncGuard, alpaka_tbb_async::Queue>;
    using Base::Base;
    void set_impl() { /**< nothing to be done */ }
    void reset_impl() { /**< nothing to be done */ }
  };

  template <>
  struct GuardTraits<alpaka_tbb_async::Queue> {
    using type = CpuTbbAsyncGuard;
  };
#else
#error "Torch guard for this backend is not defined."
#endif

  /**
   * @brief Alias for the appropriate Guard type based on the Alpaka queue.
   */
  template <typename TQueue>
  using Guard = typename GuardTraits<TQueue>::type;

}  // namespace cms::torch::alpaka

#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__ALPAKA_CONFIG_H_
