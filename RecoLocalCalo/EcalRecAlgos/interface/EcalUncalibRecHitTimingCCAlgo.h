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
#include "RecoLocalCalo/EcalRecAlgos/interface/PulseChiSqSNNLS.h"


#include "TMatrixDSym.h"
#include "TVectorD.h"


class EcalUncalibRecHitTimingCCAlgo
{
    float _startTime;
    float _stopTime;

    public:
        EcalUncalibRecHitTimingCCAlgo(const float startTime=-5, const float stopTime=5);
        ~EcalUncalibRecHitTimingCCAlgo() { };
        double computeTimeCC(const EcalDataFrame& dataFrame, const std::vector<double> &amplitudes, const EcalPedestals::Item * aped, const EcalMGPAGainRatio * aGain, const FullSampleVector &fullpulse, EcalUncalibratedRecHit& uncalibRecHit);

    private:
        FullSampleVector interpolatePulse(const FullSampleVector& fullpulse, const float t=0);
        float computeCC(const std::vector<double>& samples, const FullSampleVector& sigmalTemplate, const float& t);

};

#endif
