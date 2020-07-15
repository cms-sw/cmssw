#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

namespace cms {
  namespace cudacompat {
    thread_local dim3 blockIdx;
    thread_local dim3 gridDim;
  }  // namespace cudacompat
}  // namespace cms

namespace {
  struct InitGrid {
    InitGrid() { cms::cudacompat::resetGrid(); }
  };

  const InitGrid initGrid;

}  // namespace
