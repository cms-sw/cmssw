#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMaxSampleAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitMaxSampleAlgo_HH

/** \class EcalUncalibRecHitMaxSampleAlgo
  *  Amplitude reconstucted by the difference MAX_adc - min_adc
  *  jitter is sample number of MAX_adc, pedestal is min_adc
  *
  *  \author G. Franzoni, E. Di Marco
  */

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <class C>
class EcalUncalibRecHitMaxSampleAlgo : public EcalUncalibRecHitRecAbsAlgo<C> {
public:
  ~EcalUncalibRecHitMaxSampleAlgo() override{};
  EcalUncalibratedRecHit makeRecHit(const C& dataFrame,
                                    const double* pedestals,
                                    const double* gainRatios,
                                    const EcalWeightSet::EcalWeightMatrix** weights,
                                    const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix) override;

private:
  int16_t amplitude_, pedestal_, jitter_, sampleAdc_, gainId_;
  double chi2_;
};

/// compute rechits
template <class C>
EcalUncalibratedRecHit EcalUncalibRecHitMaxSampleAlgo<C>::makeRecHit(
    const C& dataFrame,
    const double* pedestals,
    const double* gainRatios,
    const EcalWeightSet::EcalWeightMatrix** weights,
    const EcalWeightSet::EcalChi2WeightMatrix** chi2Matrix) {
  amplitude_ = std::numeric_limits<int16_t>::min();
  pedestal_ = 4095;
  jitter_ = -1;
  chi2_ = -1;
  //bool isSaturated = 0;
  uint32_t flags = 0;
  for (int16_t iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
    gainId_ = dataFrame.sample(iSample).gainId();

    if (gainId_ == 0) {
      flags = EcalUncalibratedRecHit::kSaturated;
    }

    // ampli gain 12
    if (gainId_ == 1) {
      sampleAdc_ = dataFrame.sample(iSample).adc();
    }

    else {
      if (gainId_ == 2) {  // ampli gain 6
        sampleAdc_ = 200 + (dataFrame.sample(iSample).adc() - 200) * 2;
      } else {  // accounts for gainId_==3 or 0 - ampli gain 1 and gain0
        sampleAdc_ = 200 + (dataFrame.sample(iSample).adc() - 200) * 12;
      }
    }

    if (sampleAdc_ > amplitude_) {
      amplitude_ = sampleAdc_;
      jitter_ = iSample;
    }  // if statement

    if (sampleAdc_ < pedestal_)
      pedestal_ = sampleAdc_;

  }  // loop on samples

  return EcalUncalibratedRecHit(dataFrame.id(),
                                static_cast<double>(amplitude_ - pedestal_),
                                static_cast<double>(pedestal_),
                                static_cast<double>(jitter_ - 5),
                                chi2_,
                                flags);
}

#endif
