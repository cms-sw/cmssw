
#include <alpaka/alpaka.hpp>
#include <cstdint>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  struct KernelFilterLayerClusterByAlgo {
    template <typename TAcc>
    ALPAKA_ACC_FN void operator()(const TAcc& acc,
                                  const int32_t* layerClusterAlgoId,
                                  const int32_t* algo_number,
                                  int32_t* layerClusterMask,
                                  int32_t size) const {
      for (auto idx : alpaka::uniformElements(acc, size)) {
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
