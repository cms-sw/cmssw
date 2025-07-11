#ifndef PHYSICS_TOOLS__PYTORCH__INTERFACE__NVTX_H_
#define PHYSICS_TOOLS__PYTORCH__INTERFACE__NVTX_H_

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#include <nvtx3/nvToolsExt.h>
#endif

/**
 * @class NvtxScopedRange
 * @brief Helper class for NVTX profiling.
 *
 * Exposes a simple interface to create and manage NVTX ranges.
 * Automatically ends the range when the object goes out of scope.
 */
class NvtxScopedRange {
public:
  NvtxScopedRange(const char* msg) {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    id_ = nvtxRangeStartA(msg);
#endif
  }

  void end() {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    if (active_) {
      active_ = false;
      nvtxRangeEnd(id_);
    }
#endif
  }

  ~NvtxScopedRange() { end(); }

private:
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || \
    defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  nvtxRangeId_t id_;
  bool active_ = true;
#endif
};

#endif  // PHYSICS_TOOLS__PYTORCH__INTERFACE__NVTX_H_
