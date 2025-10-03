#ifndef PhysicsTools_PyTorchAlpaka_interface_NvtxRAII_h
#define PhysicsTools_PyTorchAlpaka_interface_NvtxRAII_h

#include <string>
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <nvtx3/nvToolsExt.h>
#endif
#include "PhysicsTools/PyTorchAlpaka/interface/Environment.h"

namespace cms::torch::alpakatools {

  using namespace cms::torchcommon;

  class NvtxRAII {
  public:
    explicit NvtxRAII(const std::string& label, const Environment env = Environment::kProduction)
        : label_(label), env_(env) {
      if (env_ >= Environment::kDevelopment)
        begin();
    }

    ~NvtxRAII() {
      if (env_ >= Environment::kDevelopment)
        end();
    }

    void begin() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      range_id_ = nvtxRangeStartA(label_.c_str());
      active_ = true;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
    }

    void end() {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
      if (active_) {
        active_ = false;
        nvtxRangeEnd(range_id_);
      }
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
    }

  private:
    const std::string label_;
    const Environment env_;
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    nvtxRangeId_t range_id_;
    bool active_ = true;
#endif  // ALPAKA_ACC_GPU_CUDA_ENABLED
  };

}  // namespace cms::torch::alpakatools

#endif  // PhysicsTools_PyTorchAlpaka_interface_NvtxRAII_h