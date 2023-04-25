#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitTimingCCAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitTimingCCAlgo_HH

/** \class EcalUncalibRecHitTimingCCAlgo
  *  CrossCorrelation algorithm for timing reconstruction
  *
  *  \author N. Minafra, J. King, C. Rogan
  */

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"

class EcalUncalibRecHitTimingCCAlgo {
public:
  EcalUncalibRecHitTimingCCAlgo(const float startTime, const float stopTime);
  double computeTimeCC(const EcalDataFrame& dataFrame,
                       const std::vector<double>& amplitudes,
                       const EcalPedestals::Item* aped,
                       const EcalMGPAGainRatio* aGain,
                       const FullSampleVector& fullpulse,
                       EcalUncalibratedRecHit& uncalibRecHit,
                       float& errOnTime,
                       const float targetTimePrecision,
                       const bool correctForOOT = true) const;

private:
  const float startTime_;
  const float stopTime_;

  static constexpr int TIME_WHEN_NOT_CONVERGING = 100;
  static constexpr int MAX_NUM_OF_ITERATIONS = 60;
  static constexpr int MIN_NUM_OF_ITERATIONS = 2;
  static constexpr float GLOBAL_TIME_SHIFT = 100;
  static constexpr float GOLDEN_RATIO = 0.61803399;
  static constexpr float ONE_MINUS_GOLDEN_RATIO = 1.0 - GOLDEN_RATIO;

  FullSampleVector interpolatePulse(const FullSampleVector& fullpulse, const float t = 0) const;
  float computeCC(const std::vector<float>& samples, const FullSampleVector& sigmalTemplate, const float t) const;
};

#endif
