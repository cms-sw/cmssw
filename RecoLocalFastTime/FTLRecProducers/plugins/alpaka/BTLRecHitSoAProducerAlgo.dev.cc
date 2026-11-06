#include <alpaka/alpaka.hpp>

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitSoA.h"
#include "DataFormats/FTLRecHitSoA/interface/BTLRecHitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "RecoLocalFastTime/Records/interface/MTDTimeCalibRecord.h"
#include "RecoLocalFastTime/FTLCommonAlgos/interface/MTDTimeCalib.h"

#include "BTLRecHitSoAProducerAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit {

  using namespace ::btlrechit;
  ALPAKA_FN_ACC float timeResolutionInNs(std::array<double, 3> tResParams, float amp) {
    return tResParams[0] * pow(amp, tResParams[1]) + tResParams[2];
  }

  ALPAKA_FN_ACC float timeWalkCorr(std::array<double, 3> twcParams, float amp) {
    // taken from SLHCUpgradeSimulations/Configuration/python/aging.py
    // for 1000 fb-1 scenario
    return twcParams[0] * pow(amp, twcParams[1]) + twcParams[2];
  }

  class BTLBaseToRecoKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  BTLBaseRecHitSoA::ConstView input,
                                  BTLRecHitSoA::View output,
                                  const double c_LYSO_,
                                  const double thresholdToKeep_,
                                  const double calibration_,
                                  const std::array<double, 2> npeSaturationCorr_,
                                  const std::array<double, 2> npeToADC_,
                                  const double npePerGeV_,
                                  const double timeCalibration_,
                                  const std::array<double, 3> tResParams_,
                                  const std::array<double, 3> twcParams_)
        const {  // when condformat for calib ready, add also tdc and qdc in inputs
      // make a strided loop over the kernel grid, covering up to "size" elements

      for (int32_t i : cms::alpakatools::uniform_elements(acc, input.metadata().size())) {
        auto entry = input[i];
        float time1Plus = entry.time1Plus();
        float time1Minus = entry.time1Minus();
        float time2Plus = entry.time2Plus();
        float time2Minus = entry.time2Minus();
        float ampPlus = entry.ampPlus();
        float ampMinus = entry.ampMinus();

        // converting the energy from ADC to energy
        ampPlus = float((float(ampPlus) - npeToADC_[0]) / npeToADC_[1]);
        ampMinus = float((float(ampMinus) - npeToADC_[0]) / npeToADC_[1]);

        //   correction for SiPM saturation (just invert the function used to model this effect in BTLElectronicsSim)
        float dR = npeSaturationCorr_[1] * npeSaturationCorr_[1] + 4. * npeSaturationCorr_[0] * ampPlus;
        ampPlus = (-npeSaturationCorr_[1] + sqrt(dR)) / (2. * (npeSaturationCorr_[0]));
        ampPlus /= npePerGeV_;
        float dL = npeSaturationCorr_[1] * npeSaturationCorr_[1] + 4. * npeSaturationCorr_[0] * ampMinus;
        ampMinus = (-npeSaturationCorr_[1] + sqrt(dL)) / (2. * (npeSaturationCorr_[0]));
        ampMinus /= npePerGeV_;

        // Apply time and energy corrections
        //   apply amp walk corrections
        auto corrR = timeWalkCorr(twcParams_, ampPlus);
        auto corrL = timeWalkCorr(twcParams_, ampMinus);
        time1Plus = time1Plus - corrR;
        time1Minus = time1Minus - corrL;
        time2Plus = time2Plus - corrR;
        time2Minus = time2Minus - corrL;

        float time1 = 0;
        float time2 = 0;
        float position = -1.;
        float position_error = -1.;
        float time_error = 0;
        float energy = 0;
        uint8_t flag = 0;

        // -- if you have both sipm info and they are not saturated
        if (entry.flagsPlus() == 0x1 && entry.flagsMinus() == 0x1) {
          time1 = 0.5f * (time1Minus + time1Plus);
          time2 = 0.5f * (time2Minus + time2Plus);  // to be discussed
          position = 0.5f * c_LYSO_ * (time1Plus - time1Minus);
          position_error = 0.6;  // as in the std btl uncalibrated hit producer
          energy = (ampPlus + ampMinus) / 2.;
          flag |= 0x3;

        }
        // --- If only one SiPM has good not saturated signal
        else if (entry.flagsMinus() == 0x1 && (entry.flagsPlus() == 0x3 || entry.flagsPlus() == 0)) {
          time1 = time1Minus;
          time2 = time2Minus;
          energy = ampMinus;
          flag |= (0x1 << 1);
        }

        else if (entry.flagsPlus() == 0x1 && (entry.flagsMinus() == 0x3 || entry.flagsMinus() == 0)) {
          time1 = time1Plus;
          time2 = time2Plus;
          energy = ampPlus;
          flag |= (0x1 << 1);
        }

        // energy calibration
        energy *= calibration_;

        // --- Time calibration: for the time being just removes a time offset in BTL
        time1 -= timeCalibration_;

        time_error = timeResolutionInNs(tResParams_, energy);

        // Now fill flags
        // good is 1--> 2 channels && over threshold, bad is 0
        if (energy > thresholdToKeep_ && flag == 0x3) {
          flag = 1;
        } else {
          flag = 0;
        }

#ifdef EDM_ML_DEBUG
        printf("RecHit SoA with raw id %i \n", entry.detId().rawId());
        printf("Calibrations: timeCal = %f \n", timeCalibration_);
        printf("              energyCal = %f \n", calibration_);
        printf("Time 1  -,+ (%f, %f). Time 1 = %f +/- %f \n", time1Minus, time1Plus, time1, time_error);
        printf("Time 2  -,+ (%f, %f). Time 2 = %f \n", time2Minus, time2Plus, time2);
        printf("Energy  -,+ (%f, %f). Energy = %f \n", ampMinus, ampPlus, energy);
        printf("Position and error (%f, %f) \n", position, position_error);

#endif

        // fill the rechit
        output[i] = {entry.detId(),
                     entry.row(),  //dummy
                     time1,
                     time2,
                     energy,
                     position,
                     time_error,
                     position_error,  // dummy
                     flag};
      }
    }
  };

  void BTLRecHitSoAProducerAlgo::fromBaseToReco(Queue& queue,
                                                BTLBaseRecHitSoA::ConstView const& input,
                                                BTLRecHitSoA::View& output,
                                                const double c_LYSO_,
                                                const double thresholdToKeep_,
                                                const double calibration_,
                                                const std::array<double, 2> npeSaturationCorr_,
                                                const std::array<double, 2> npeToADC_,
                                                const double npePerGeV_,
                                                const double timeCalibration_,
                                                const std::array<double, 3> tResParams_,
                                                const std::array<double, 3> twcParams_) {
    // Use 64 items per group.
    // This value is arbitrary, but it's a reasonable starting point.
    uint32_t items = 64;

    // Use as many groups as needed to cover the whole problem.
    // If this value is too large, a smaller number of blocks can give better performance.
    uint32_t groups = cms::alpakatools::divide_up_by(input.metadata().size(), items);

    auto grid = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue,
                        grid,
                        BTLBaseToRecoKernel{},
                        input,
                        output,
                        c_LYSO_,
                        thresholdToKeep_,
                        calibration_,
                        npeSaturationCorr_,
                        npeToADC_,
                        npePerGeV_,
                        timeCalibration_,
                        tResParams_,
                        twcParams_);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::btlrechit
