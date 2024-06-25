#ifndef HeterogeneousCore_AlpakaInterface_interface_HostOnlyTask_h
#define HeterogeneousCore_AlpakaInterface_interface_HostOnlyTask_h

#include <functional>
#include <memory>

#include <fmt/format.h>

#include <alpaka/alpaka.hpp>

namespace alpaka {

  //! A task that is guaranted not to call any GPU-ralated APIs
  //!
  //! These tasks can be enqueued directly to the native GPU queues, without the use of a
  //! dedicated host-side worker thread.
  class HostOnlyTask {
  public:
    HostOnlyTask(std::function<void(std::exception_ptr)> task) : task_(std::move(task)) {}

    void operator()(std::exception_ptr eptr) const { task_(eptr); }

  private:
    std::function<void(std::exception_ptr)> task_;
  };

  namespace trait {

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    //! The CUDA async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueCudaRtNonBlocking, HostOnlyTask> {
      using TApi = ApiCudaRt;

      static void CUDART_CB callback(cudaStream_t queue, cudaError_t status, void* arg) {
        std::unique_ptr<HostOnlyTask> pTask(static_cast<HostOnlyTask*>(arg));
        if (status == cudaSuccess) {
          (*pTask)(nullptr);
        } else {
          // wrap the exception in a try-catch block to let GDB "catch throw" break on it
          try {
            throw std::runtime_error(fmt::format("CUDA error: callback of stream {} received error {}: {}.",
                                                 fmt::ptr(queue),
                                                 cudaGetErrorName(status),
                                                 cudaGetErrorString(status)));
          } catch (std::exception&) {
            // pass the exception to the task
            (*pTask)(std::current_exception());
          }
        }
      }

      ALPAKA_FN_HOST static auto enqueue(QueueCudaRtNonBlocking& queue, HostOnlyTask task) -> void {
        auto pTask = std::make_unique<HostOnlyTask>(std::move(task));
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            cudaStreamAddCallback(alpaka::getNativeHandle(queue), callback, static_cast<void*>(pTask.release()), 0u));
      }
    };
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
    //! The HIP async queue enqueue trait specialization for "safe tasks"
    template <>
    struct Enqueue<QueueHipRtNonBlocking, HostOnlyTask> {
      using TApi = ApiHipRt;

      static void callback(hipStream_t queue, hipError_t status, void* arg) {
        std::unique_ptr<HostOnlyTask> pTask(static_cast<HostOnlyTask*>(arg));
        if (status == hipSuccess) {
          (*pTask)(nullptr);
        } else {
          // wrap the exception in a try-catch block to let GDB "catch throw" break on it
          try {
            throw std::runtime_error(fmt::format("HIP error: callback of stream {} received error {}: {}.",
                                                 fmt::ptr(queue),
                                                 hipGetErrorName(status),
                                                 hipGetErrorString(status)));
          } catch (std::exception&) {
            // pass the exception to the task
            (*pTask)(std::current_exception());
          }
        }
      }

      ALPAKA_FN_HOST static auto enqueue(QueueHipRtNonBlocking& queue, HostOnlyTask task) -> void {
        auto pTask = std::make_unique<HostOnlyTask>(std::move(task));
        ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
            hipStreamAddCallback(alpaka::getNativeHandle(queue), callback, static_cast<void*>(pTask.release()), 0u));
      }
    };
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

  }  // namespace trait

}  // namespace alpaka

#endif  // HeterogeneousCore_AlpakaInterface_interface_HostOnlyTask_h
