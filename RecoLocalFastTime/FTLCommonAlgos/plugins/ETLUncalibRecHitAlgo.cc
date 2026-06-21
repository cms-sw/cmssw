#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "ETLUncalibRecHitAlgo.h"

FTLUncalibratedRecHit ETLUncalibRecHitAlgo::makeRecHit(const ETLDataFrame& dataFrame) const {
  constexpr int iSample = 2;  //only in-time sample
  const auto& sample = dataFrame.sample(iSample);

  double time = double(sample.toa()) * toaLSBToNS_;
  double time_over_threshold = double(sample.tot()) * toaLSBToNS_;
  const std::array<double, 1> time_over_threshold_V = {{time_over_threshold}};

  unsigned char flag = 0;

  LogDebug("ETLUncalibRecHit") << "ADC+: set the charge to: " << time_over_threshold << ' ' << sample.tot() << ' '
                               << toaLSBToNS_;

  if (time_over_threshold == 0) {
    LogDebug("ETLUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() << ' ' << toaLSBToNS_;

  } else {
    // Time-walk correction for toa
    double timeWalkCorr = timeCorr_p0_ + timeCorr_p1_ * time_over_threshold +
                          timeCorr_p2_ * time_over_threshold * time_over_threshold +
                          timeCorr_p3_ * time_over_threshold * time_over_threshold * time_over_threshold;

    time -= timeWalkCorr;

    LogDebug("ETLUncalibRecHit") << "ADC+: set the time to: " << time << ' ' << sample.toa() << ' ' << toaLSBToNS_
                                 << " .Timewalk correction: " << timeWalkCorr;
  }

  LogDebug("ETLUncalibRecHit") << "Final uncalibrated time_over_threshold: " << time_over_threshold;

  const std::array<double, 1> emptyV = {{0.}};

  double timeError = timeError_.evaluate(time_over_threshold_V, emptyV);

  return FTLUncalibratedRecHit(dataFrame.id(),
                               dataFrame.row(),
                               dataFrame.column(),
                               {time_over_threshold_V[0], 0.f},
                               {time, 0.f},
                               timeError,
                               -1.f,
                               -1.f,
                               flag);
}

void ETLUncalibRecHitAlgo::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<uint32_t>("adcNbits");
  desc.add<double>("adcSaturation");
  desc.add<double>("toaLSB_ns");
  desc.add<std::string>("timeResolutionInNs");
  desc.add<double>("timeCorr_p0");
  desc.add<double>("timeCorr_p1");
  desc.add<double>("timeCorr_p2");
  desc.add<double>("timeCorr_p3");
}
