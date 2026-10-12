#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/BTLRecHitsErrorEstimatorIM.h"

#include "BTLUncalibRecHitAlgo.h"

FTLUncalibratedRecHit BTLUncalibRecHitAlgo::makeRecHit(const BTLDataFrame& dataFrame) const {
  // The reconstructed amplitudes and times of the right and left hits are saved in a std::pair
  std::pair<double, double> amplitude(0., 0.);
  std::pair<double, double> time(0., 0.);

  unsigned char flag = 0;

  const auto& sampleRight = dataFrame.sample(0);
  const auto& sampleLeft = dataFrame.sample(1);

  double nHits = 0.;

  LogDebug("BTLUncalibRecHit") << "Original input time t1, t2 " << double(sampleRight.toa()) * tdc_to_ns_ << ", "
                               << double(sampleLeft.toa()) * tdc_to_ns_ << std::endl;

  // --- Reconstruct amplitude and time of the crystal's right channel
  if (sampleRight.data() > 0) {
    // Convert ADC counts to MeV and TDC counts to ns
    amplitude.first = (double(sampleRight.data()) - npeToADC_[0]) / npeToADC_[1];
    time.first = double(sampleRight.toa()) * tdc_to_ns_;

    // Correction for SiPM saturation (just invert the function used to model this effect in BTLElectronicsSim)
    float d = npeSaturationCorr_[1] * npeSaturationCorr_[1] + 4. * npeSaturationCorr_[0] * amplitude.first;
    amplitude.first = (-npeSaturationCorr_[1] + sqrt(d)) / (2. * (npeSaturationCorr_[0]));
    amplitude.first /= npePerMeV_;

    // Correct the time of the right SiPM for the time-walk
    time.first -= timeWalkCorr_.evaluate(std::array<double, 1>{{amplitude.first}}, std::array<double, 1>{{0.0}});

    flag |= 0x1;
    nHits += 1.;

    // Set bits 4 (ADC) and 5 (TDC) of flag when the right channel's ADC or TDC is saturated
    if (sampleRight.data() == BTLSample::kDataMask)
      flag |= (0x1 << 4);
    if (sampleRight.toa() == BTLSample::kToAMask)
      flag |= (0x1 << 5);
  }

  // --- Reconstruct amplitude and time of the crystal's left channel
  if (sampleLeft.data() > 0) {
    // Convert ADC counts to MeV and TDC counts to ns
    amplitude.second = (double(sampleLeft.data()) - npeToADC_[0]) / npeToADC_[1];
    time.second = double(sampleLeft.toa()) * tdc_to_ns_;

    // Correction for SiPM saturation (just invert the function used to model this effect in BTLElectronicsSim)
    float d = npeSaturationCorr_[1] * npeSaturationCorr_[1] + 4. * npeSaturationCorr_[0] * amplitude.second;
    amplitude.second = (-npeSaturationCorr_[1] + sqrt(d)) / (2. * (npeSaturationCorr_[0]));
    amplitude.second /= npePerMeV_;

    // Correct the time of the left SiPM for the time-walk
    time.second -= timeWalkCorr_.evaluate(std::array<double, 1>{{amplitude.second}}, std::array<double, 1>{{0.0}});

    flag |= (0x1 << 1);
    nHits += 1.;

    // Set bits 6 (ADC) and 7 (TDC) of flag when the left channel's ADC or TDC is saturated
    if (sampleLeft.data() == BTLSample::kDataMask)
      flag |= (0x1 << 6);
    if (sampleLeft.toa() == BTLSample::kToAMask)
      flag |= (0x1 << 7);
  }

  // --- Calculate the error on the hit time using the provided parameterization

  const std::array<double, 1> amplitudeV = {{(amplitude.first + amplitude.second) / nHits}};
  const std::array<double, 1> emptyV = {{0.}};

  double timeError = (nHits > 0. ? timeError_.evaluate(amplitudeV, emptyV) : -1.);

  // Calculate the position
  // Distance from center of bar to hit

  double position = 0.5f * (c_LYSO_ * (time.second - time.first));
  double positionError = BTLRecHitsErrorEstimatorIM::positionError();

  LogDebug("BTLUncalibRecHit") << "DetId: " << dataFrame.id().rawId() << " x position = " << position << " +/- "
                               << positionError;
  LogDebug("BTLUncalibRecHit") << "ADC+: set the charge to: (" << amplitude.first << ", " << amplitude.second << ")  ("
                               << sampleRight.data() << ", " << sampleLeft.data() << ") " << invADCPerMeV_ << ' '
                               << std::endl;
  LogDebug("BTLUncalibRecHit") << "TDC+: set the time to: (" << time.first << ", " << time.second << ")  ("
                               << sampleRight.toa() << ", " << sampleLeft.toa() << ") " << tdc_to_ns_ << ' '
                               << std::endl;

  return FTLUncalibratedRecHit(
      dataFrame.id(), dataFrame.row(), dataFrame.column(), amplitude, time, timeError, position, positionError, flag);
}

void BTLUncalibRecHitAlgo::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<double>("invLightSpeedLYSO");
  desc.add<std::vector<double>>("npeToADC");
  desc.add<double>("npePerMeV");
  desc.add<std::vector<double>>("npeSaturationCorrection");
  desc.add<double>("tdcLSB_ns");
  desc.add<std::string>("timeResolutionInNs");
  desc.add<std::string>("timeWalkCorrection");
}
