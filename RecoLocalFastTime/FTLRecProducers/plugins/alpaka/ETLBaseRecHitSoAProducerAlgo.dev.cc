#include <cstdio>

#include <alpaka/alpaka.hpp>

#include "DataFormats/FTLRecHitSoA/interface/alpaka/ETLBaseRecHitDeviceCollection.h"
#include "DataFormats/FTLDigiSoA/interface/alpaka/ETLDigiDeviceCollection.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "ETLBaseRecHitSoAProducerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit {

  using namespace ::etlrechit;

  class ETLdigiToBaseKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::etldigi::ETLDigiSoA::ConstView input,
                                  ETLBaseRecHitSoA::View output,
                                  const uint32_t adcNBits_,
                                  const double adcSaturation_,
                                  const double adcLSB_,
                                  const double toaLSB_ns_,
                                  const double tdcWindowStart_) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : cms::alpakatools::uniform_elements(acc, input.metadata().size())) {
        auto entry = input[i];
        // here you should call your functions to apply TDC and QDC, reading timecoarse, fine, ... etc from digi

        // for the times at first and second th, still in clock units
        // atm tdc and qdc calibs are fixed to dummy values for each channel, hence rawId, ch, and the bool to select branch 1 or 2 are not used.
        float time = 12.5 - entry.ToAdata() * toaLSB_ns_ + tdcWindowStart_;
        float time_over_threshold = entry.ToTdata() * toaLSB_ns_;

        // detId from rawId
        DetId detId(entry.rawId());

        uint8_t row = entry.rowID();
        uint8_t col = entry.colID();

#ifdef EDM_ML_DEBUG
        printf("Base recHit SoA with raw id %i \n", entry.rawId());
        printf("ToA before corrections: %f \n", time);
        printf("ToT before corrections: %f \n", time_over_threshold);
#endif

        // fill the base rechit
        output[i] = {
            detId,
            row,
            col,
            time,  // in ns
            time_over_threshold,
            0,  // flags
        };
      }
    }
  };

  void ETLBaseRecHitSoAProducerAlgo::fromDigiToBase(Queue& queue,
                                                    ::etldigi::ETLDigiSoA::ConstView const& input,
                                                    ETLBaseRecHitSoA::View& output,
                                                    const uint32_t adcNBits_,
                                                    const double adcSaturation_,
                                                    const double adcLSB_,
                                                    const double toaLSB_ns_,
                                                    const double tdcWindowStart_) {
    // Use 64 items per group.
    // This value is arbitrary, but it's a reasonable starting point.
    uint32_t items = 64;

    // Use as many groups as needed to cover the whole problem.
    // If this value is too large, a smaller number of blocks can give better performance.
    uint32_t groups = cms::alpakatools::divide_up_by(input.metadata().size(), items);

    auto grid = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        ETLdigiToBaseKernel{},
                        input,
                        output,
                        adcNBits_,
                        adcSaturation_,
                        adcLSB_,
                        toaLSB_ns_,
                        tdcWindowStart_);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit
