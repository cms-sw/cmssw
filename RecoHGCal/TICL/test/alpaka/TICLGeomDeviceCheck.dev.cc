#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TICLGeomDeviceCheck.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace ticlgeomtest {

    class CheckLookupKernel {
    public:
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    TICLGeomCommonSoAConstView common,
                                    TICLGeomLookupSoAConstView lookup,
                                    int32_t* nBad) const {
        for (auto i : cms::alpakatools::uniform_elements(acc, common.metadata().size())) {
          const uint32_t rawDetId = common[i].rawDetId();
          if (ticlgeom::indexOf(common, rawDetId) != static_cast<int32_t>(i) ||
              ticlgeom::denseIdOf(lookup, common, rawDetId) != static_cast<int32_t>(i)) {
            alpaka::atomicAdd(acc, nBad, 1, alpaka::hierarchy::Blocks{});
          }
        }
      }
    };

    int32_t checkLookup(Queue& queue, TICLGeomCommonSoAConstView common, TICLGeomLookupSoAConstView lookup) {
      auto nBad = cms::alpakatools::make_device_buffer<int32_t>(queue);
      alpaka::memset(queue, nBad, 0);

      const auto workDiv =
          cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(common.metadata().size(), 256), 256);
      alpaka::exec<Acc1D>(queue, workDiv, CheckLookupKernel{}, common, lookup, nBad.data());

      auto nBadHost = cms::alpakatools::make_host_buffer<int32_t>(queue);
      alpaka::memcpy(queue, nBadHost, nBad);
      alpaka::wait(queue);
      return *nBadHost;
    }

  }  // namespace ticlgeomtest

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
