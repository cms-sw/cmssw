#include "testESAlgoAsync.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  AlpakaESTestDataDDevice testESAlgoAsync(Queue& queue,
                                          AlpakaESTestDataADevice const& dataA,
                                          cms::alpakatest::AlpakaESTestDataB<Device> const& dataB) {
    auto const size = std::min(dataA->metadata().size(), static_cast<int>(dataB.size()));
    AlpakaESTestDataDDevice ret(size, queue);

    auto const& deviceProperties = alpaka::getAccDevProps<Acc1D>(alpaka::getDev(queue));
    uint32_t maxThreadsPerBlock = deviceProperties.m_blockThreadExtentMax[0];

    uint32_t threadsPerBlock = maxThreadsPerBlock;
    uint32_t blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    uint32_t elementsPerThread = 1;
    auto workDiv = WorkDiv1D{blocksPerGrid, threadsPerBlock, elementsPerThread};

    alpaka::exec<Acc1D>(
        queue,
        workDiv,
        [] ALPAKA_FN_ACC(Acc1D const& acc,
                         AlpakaESTestDataADevice::ConstView a,
                         int const* b,
                         AlpakaESTestDataDDevice::View c,
                         int size) {
          const int32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
          const int32_t stride = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];
          for (auto i = thread; i < size; i += stride) {
            c[i] = a.z()[i] + b[i];
          }
        },
        dataA.view(),
        dataB.data(),
        ret.view(),
        size);

    return ret;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
