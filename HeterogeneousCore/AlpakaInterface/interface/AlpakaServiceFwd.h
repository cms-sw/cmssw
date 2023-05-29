#ifndef HeterogeneousCore_AlpakaInterface_interface_AlpakaServiceFwd_h
#define HeterogeneousCore_AlpakaInterface_interface_AlpakaServiceFwd_h

// Forward declaration of the alpaka accelerator namespaces and of the AlpakaService for each of them.
//
// This file is under HeterogeneousCore/AlpakaInterface to avoid introducing a dependency on
// HeterogeneousCore/AlpakaServices and HeterogeneousCore/AlpakaCore.

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
namespace alpaka_cuda_async {
  class AlpakaService;
}  // namespace alpaka_cuda_async
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
namespace alpaka_rocm_async {
  class AlpakaService;
}  // namespace alpaka_rocm_async
#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
namespace alpaka_serial_sync {
  class AlpakaService;
}  // namespace alpaka_serial_sync
#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
namespace alpaka_tbb_async {
  class AlpakaService;
}  // namespace alpaka_tbb_async
#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#endif  // HeterogeneousCore_AlpakaInterface_interface_AlpakaServiceFwd_h
