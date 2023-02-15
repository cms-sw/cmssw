#ifndef HeterogeneousCore_AlpakaInterface_interface_config_h
#define HeterogeneousCore_AlpakaInterface_interface_config_h

#include <type_traits>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"

namespace alpaka_common {

  // common types and dimensions
  using Idx = uint32_t;
  using Extent = uint32_t;
  using Offsets = Extent;

  using Dim0D = alpaka::DimInt<0u>;
  using Dim1D = alpaka::DimInt<1u>;
  using Dim2D = alpaka::DimInt<2u>;
  using Dim3D = alpaka::DimInt<3u>;

  template <typename TDim>
  using Vec = alpaka::Vec<TDim, Idx>;
  using Vec1D = Vec<Dim1D>;
  using Vec2D = Vec<Dim2D>;
  using Vec3D = Vec<Dim3D>;
  using Scalar = Vec<Dim0D>;

  template <typename TDim>
  using WorkDiv = alpaka::WorkDivMembers<TDim, Idx>;
  using WorkDiv1D = WorkDiv<Dim1D>;
  using WorkDiv2D = WorkDiv<Dim2D>;
  using WorkDiv3D = WorkDiv<Dim3D>;

  // host types
  using DevHost = alpaka::DevCpu;
  using PltfHost = alpaka::Pltf<DevHost>;

}  // namespace alpaka_common

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
namespace alpaka_cuda_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCudaRt;
  using Device = alpaka::DevCudaRt;
  using Queue = alpaka::QueueCudaRtNonBlocking;
  using Event = alpaka::EventCudaRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuCudaRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_cuda_async

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_cuda_async
#define ALPAKA_TYPE_SUFFIX CudaAsync
#endif

#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED
namespace alpaka_rocm_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfHipRt;
  using Device = alpaka::DevHipRt;
  using Queue = alpaka::QueueHipRtNonBlocking;
  using Event = alpaka::EventHipRt;

  template <typename TDim>
  using Acc = alpaka::AccGpuHipRt<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_rocm_async

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_rocm_async
#define ALPAKA_TYPE_SUFFIX ROCmAsync
#endif

#endif  // ALPAKA_ACC_GPU_HIP_ENABLED

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
namespace alpaka_serial_sync {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuSerial<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_serial_sync

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_serial_sync
#define ALPAKA_TYPE_SUFFIX SerialSync
#endif

#endif  // ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
namespace alpaka_tbb_async {
  using namespace alpaka_common;

  using Platform = alpaka::PltfCpu;
  using Device = alpaka::DevCpu;
  using Queue = alpaka::QueueCpuNonBlocking;
  using Event = alpaka::EventCpu;

  template <typename TDim>
  using Acc = alpaka::AccCpuTbbBlocks<TDim, Idx>;
  using Acc1D = Acc<Dim1D>;
  using Acc2D = Acc<Dim2D>;
  using Acc3D = Acc<Dim3D>;

}  // namespace alpaka_tbb_async

#ifdef ALPAKA_ACCELERATOR_NAMESPACE
#define ALPAKA_DUPLICATE_NAMESPACE
#else
#define ALPAKA_ACCELERATOR_NAMESPACE alpaka_tbb_async
#define ALPAKA_TYPE_SUFFIX TbbAsync
#endif

#endif  // ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#if defined ALPAKA_DUPLICATE_NAMESPACE
#error Only one alpaka backend symbol can be defined at the same time: ALPAKA_ACC_GPU_CUDA_ENABLED, ALPAKA_ACC_GPU_HIP_ENABLED, ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED, ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED.
#endif

#if defined ALPAKA_ACCELERATOR_NAMESPACE

// create a new backend-specific identifier based on the original type name and a backend-specific suffix
#define ALPAKA_TYPE_ALIAS__(TYPE, SUFFIX) TYPE##SUFFIX
#define ALPAKA_TYPE_ALIAS_(TYPE, SUFFIX) ALPAKA_TYPE_ALIAS__(TYPE, SUFFIX)
#define ALPAKA_TYPE_ALIAS(TYPE) ALPAKA_TYPE_ALIAS_(TYPE, ALPAKA_TYPE_SUFFIX)

// declare the backend-specific identifier as an alias for the namespace-based type name
#define DECLARE_ALPAKA_TYPE_ALIAS(TYPE) using ALPAKA_TYPE_ALIAS(TYPE) = ALPAKA_ACCELERATOR_NAMESPACE::TYPE

// define a null-terminated string containing the backend-specific identifier
#define ALPAKA_TYPE_ALIAS_NAME(TYPE) EDM_STRINGIZE(ALPAKA_TYPE_ALIAS(TYPE))

#endif  // ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HeterogeneousCore_AlpakaInterface_interface_config_h
