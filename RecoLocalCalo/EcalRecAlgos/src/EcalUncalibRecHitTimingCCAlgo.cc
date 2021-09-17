#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitTimingCCAlgo.h"

EcalUncalibRecHitTimingCCAlgo::EcalUncalibRecHitTimingCCAlgo(const float startTime,
                                                             const float stopTime,
                                                             const float targetTimePrecision)
    : startTime_(startTime), stopTime_(stopTime), targetTimePrecision_(targetTimePrecision) {}

double EcalUncalibRecHitTimingCCAlgo::computeTimeCC(const EcalDataFrame& dataFrame,
                                                    const std::vector<double>& amplitudes,
                                                    const EcalPedestals::Item* aped,
                                                    const EcalMGPAGainRatio* aGain,
                                                    const FullSampleVector& fullpulse,
                                                    EcalUncalibratedRecHit& uncalibRecHit,
                                                    float& errOnTime) const {
  constexpr unsigned int nsample = EcalDataFrame::MAXSAMPLES;

  double maxamplitude = -std::numeric_limits<double>::max();
  float pulsenorm = 0.;

  std::vector<float> pedSubSamples(nsample);
  for (unsigned int iSample = 0; iSample < nsample; iSample++) {
    const EcalMGPASample& sample = dataFrame.sample(iSample);

    float amplitude = 0.;
    int gainId = sample.gainId();

    double pedestal = 0.;
    double gainratio = 1.;

    if (gainId == 0 || gainId == 3) {
      pedestal = aped->mean_x1;
      gainratio = aGain->gain6Over1() * aGain->gain12Over6();
    } else if (gainId == 1) {
      pedestal = aped->mean_x12;
      gainratio = 1.;
    } else if (gainId == 2) {
      pedestal = aped->mean_x6;
      gainratio = aGain->gain12Over6();
    }

    amplitude = (static_cast<float>(sample.adc()) - pedestal) * gainratio;

    if (gainId == 0) {
      //saturation
      amplitude = (4095. - pedestal) * gainratio;
    }

    pedSubSamples[iSample] = amplitude;

    if (amplitude > maxamplitude) {
      maxamplitude = amplitude;
    }
    pulsenorm += fullpulse(iSample);
  }

  int ipulse = -1;
  for (auto const& amplit : amplitudes) {
    ipulse++;
    int bxp3 = ipulse - 2;
    int firstsamplet = std::max(0, bxp3);
    int offset = 7 - bxp3;

    for (unsigned int isample = firstsamplet; isample < nsample; ++isample) {
      auto const pulse = fullpulse(isample + offset);
      pedSubSamples[isample] = std::max(0., pedSubSamples[isample] - amplit * pulse / pulsenorm);
    }
  }

  // Start of time computation
  float tStart = startTime_ + GLOBAL_TIME_SHIFT;
  float tStop = stopTime_ + GLOBAL_TIME_SHIFT;
  float tM = (tStart + tStop) / 2;

  float distStart, distStop;
  int counter = 0;

  do {
    ++counter;
    distStart = computeCC(pedSubSamples, fullpulse, tStart);
    distStop = computeCC(pedSubSamples, fullpulse, tStop);

    if (distStart > distStop) {
      tStop = tM;
    } else {
      tStart = tM;
    }
    tM = (tStart + tStop) / 2;

  } while (tStop - tStart > targetTimePrecision_ && counter < MAX_NUM_OF_ITERATIONS);

  tM -= GLOBAL_TIME_SHIFT;
  errOnTime = targetTimePrecision_;

  if (counter < MIN_NUM_OF_ITERATIONS || counter > MAX_NUM_OF_ITERATIONS - 1) {
    tM = TIME_WHEN_NOT_CONVERGING * ecalPh1::Samp_Period;
    //Negative error means that there was a problem with the CC
    errOnTime = -targetTimePrecision_ / ecalPh1::Samp_Period;
  }

  return -tM / ecalPh1::Samp_Period;
}

FullSampleVector EcalUncalibRecHitTimingCCAlgo::interpolatePulse(const FullSampleVector& fullpulse,
                                                                 const float time) const {
  // t is in ns
  int shift = time / ecalPh1::Samp_Period;
  if (time < 0)
    shift -= 1;
  float tt = time / ecalPh1::Samp_Period - shift;

  FullSampleVector interpPulse;
  // 2nd poly with avg
  unsigned int numberOfSamples = fullpulse.size();
  auto facM1orP2 = 0.25 * tt * (tt - 1);
  auto fac = (0.25 * (tt - 2) - 0.5 * (tt + 1)) * (tt - 1);
  auto facP1 = (0.25 * (tt + 1) - 0.5 * (tt - 2)) * tt;
  for (unsigned int i = 1; i < numberOfSamples - 2; ++i) {
    float a =
        facM1orP2 * fullpulse[i - 1] + fac * fullpulse[i] + facP1 * fullpulse[i + 1] + facM1orP2 * fullpulse[i + 2];
    if (a > 0)
      interpPulse[i] = a;
    else
      interpPulse[i] = 0;
  }
  interpPulse[0] = facM1orP2 * fullpulse[0] + facP1 * fullpulse[1] + facM1orP2 * fullpulse[2];
  interpPulse[numberOfSamples - 2] = facM1orP2 * fullpulse[numberOfSamples - 3] + fac * fullpulse[numberOfSamples - 2] +
                                     facP1 * fullpulse[numberOfSamples - 1];
  interpPulse[numberOfSamples - 1] = 2 * facM1orP2 * fullpulse[numberOfSamples - 2] -
                                     4 * facM1orP2 * fullpulse[numberOfSamples - 1] +
                                     facP1 * fullpulse[numberOfSamples - 1];

  FullSampleVector interpPulseShifted;
  for (int i = 0; i < interpPulseShifted.size(); ++i) {
    if (i + shift >= 0 && i + shift < interpPulse.size())
      interpPulseShifted[i] = interpPulse[i + shift];
    else
      interpPulseShifted[i] = 0;
  }
  return interpPulseShifted;
}

float EcalUncalibRecHitTimingCCAlgo::computeCC(const std::vector<float>& samples,
                                               const FullSampleVector& signalTemplate,
                                               const float time) const {
  constexpr int exclude = 1;
  float powerSamples = 0.;
  float powerTemplate = 0.;
  float cc = 0.;
  auto interpolated = interpolatePulse(signalTemplate, time);
  for (int i = exclude; i < int(samples.size() - exclude); ++i) {
    powerSamples += std::pow(samples[i], 2);
    powerTemplate += std::pow(interpolated[i], 2);
    cc += interpolated[i] * samples[i];
  }

  float denominator = std::sqrt(powerTemplate * powerSamples);
  return cc / denominator;
}
