#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "verifyDeviceObjectAsync.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  cms::alpakatools::host_buffer<bool> verifyDeviceObjectAsync(Queue& queue,
                                                              portabletest::TestDeviceObject const& deviceObject) {
    auto tmp = cms::alpakatools::make_device_buffer<bool>(queue);
    alpaka::exec<Acc1D>(
        queue,
        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
        [] ALPAKA_FN_ACC(Acc1D const& acc, portabletest::TestStruct const* obj, bool* result) {
          if (cms::alpakatools::once_per_grid(acc)) {
            *result = (obj->x == 6. and obj->y == 14. and obj->z == 15. and obj->id == 52);
          }
        },
        deviceObject.data(),
        tmp.data());
    auto ret = cms::alpakatools::make_host_buffer<bool>(queue);
    alpaka::memcpy(queue, ret, tmp);
    return ret;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
