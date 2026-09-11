#include <alpaka/alpaka.hpp>

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitSoA.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoLocalFastTime/Records/interface/MTDTimeCalibRecord.h"
#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

#include "ETLRecHitSoAProducerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit {

  using namespace ::etlrechit;

  class ETLBaseToRecoKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ETLBaseRecHitSoA::ConstView input,
                                  ETLRecHitSoA::View output,
                                  const double thresholdToKeep_,
                                  const double calibration_,
                                  const double timeResInNs_,
                                  const double timeCorr_p0_,
                                  const double timeCorr_p2_,
                                  const double timeCorr_p1_,
                                  const double timeCorr_p3_) const {
      // make a strided loop over the kernel grid, covering up to "size" elements

      for (int32_t i : cms::alpakatools::uniform_elements(acc, input.metadata().size())) {
        auto entry = input[i];
        float toa = 0;
        float tot = 0;
        float time_error = 0;
        float energy = -1;  // dummy
        uint8_t flag = 0;   // assumed to be ok

        //!!!!!!! position error calculation to be added

        // time set
        toa = entry.toa();
        tot = entry.tot();

        // Time-walk correction for toa
        float timeWalkCorr =
            timeCorr_p0_ + timeCorr_p1_ * tot + timeCorr_p2_ * tot * tot + timeCorr_p3_ * tot * tot * tot;
        toa -= timeWalkCorr;

        // --- Energy calibration
        energy = tot;            //for ETL, it is the time_over_threshold
        energy *= calibration_;  // in GeV

        time_error = timeResInNs_;

        if (energy > thresholdToKeep_) {
          flag = 1;
        } else {
          flag = 0;
        }

#ifdef EDM_ML_DEBUG

        printf("RecHit SoA with raw id %i \n", entry.detId().rawId());
        printf("Time of arrival: %f +- %f \n", toa, time_error);
        printf("Time over threshold: %f \n", tot);
        printf("Energy %f \n", energy);
        printf("Position: %f +- %f \n", position, position_error);

#endif

        // fill the rechit
        output[i] = {entry.detId(), entry.row(), entry.column(), toa, tot, time_error, flag};
      }
    }
  };

  void ETLRecHitSoAProducerAlgo::fromBaseToReco(Queue& queue,
                                                ETLBaseRecHitSoA::ConstView const& input,
                                                ETLRecHitSoA::View& output,
                                                const double thresholdToKeep_,
                                                const double calibration_,
                                                const double timeResInNs_,
                                                const double timeCorr_p0_,
                                                const double timeCorr_p2_,
                                                const double timeCorr_p1_,
                                                const double timeCorr_p3_) {
    // Use 64 items per group.
    // This value is arbitrary, but it's a reasonable starting point.
    uint32_t items = 64;

    // Use as many groups as needed to cover the whole problem.
    // If this value is too large, a smaller number of blocks can give better performance.
    uint32_t groups = cms::alpakatools::divide_up_by(input.metadata().size(), items);

    auto grid = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        ETLBaseToRecoKernel{},
                        input,
                        output,
                        thresholdToKeep_,
                        calibration_,
                        timeResInNs_,
                        timeCorr_p0_,
                        timeCorr_p2_,
                        timeCorr_p1_,
                        timeCorr_p3_);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::etlrechit
