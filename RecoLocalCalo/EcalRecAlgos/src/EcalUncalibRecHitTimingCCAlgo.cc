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
                                                    float& errOnTime) {
  constexpr unsigned int nsample = EcalDataFrame::MAXSAMPLES;

  double maxamplitude = -std::numeric_limits<double>::max();

  double pulsenorm = 0.;

  std::vector<double> pedSubSamples(nsample);
  for (unsigned int iSample = 0; iSample < nsample; iSample++) {
    const EcalMGPASample& sample = dataFrame.sample(iSample);

    double amplitude = 0.;
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

    amplitude = ((double)(sample.adc()) - pedestal) * gainratio;

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

  std::vector<double>::const_iterator amplit;
  for (amplit = amplitudes.begin(); amplit < amplitudes.end(); ++amplit) {
    int ipulse = std::distance(amplitudes.begin(), amplit);
    // The following 3 lines are copied from EcalRecAlgos/interface/EcalUncalibRecHitTimeWeightsAlgo.h
    int bx = ipulse - 5;
    int firstsamplet = std::max(0, bx + 3);
    int offset = 7 - 3 - bx;

    std::vector<double> pulse(nsample);
    for (unsigned int isample = firstsamplet; isample < nsample; ++isample) {
      pulse.at(isample) = fullpulse(isample + offset);
      pedSubSamples.at(isample) =
          std::max(0., pedSubSamples.at(isample) - amplitudes[ipulse] * pulse.at(isample) / pulsenorm);
    }
  }

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

  if (counter < MIN_NUM_OF_ITERATIONS || counter > MAX_NUM_OF_ITERATIONS - 1) {
    if (counter > MAX_NUM_OF_ITERATIONS / 2)
      //Produce a log if minimization took too long
      edm::LogInfo("EcalUncalibRecHitTimingCCAlgo::computeTimeCC")
          << "Minimization Counter too high: " << counter << std::endl;
    tM = TIME_WHEN_NOT_CONVERGING * ecalPh1::Samp_Period;
    //Negative error means that there was a problem with the CC
    errOnTime = -targetTimePrecision_ / ecalPh1::Samp_Period;
  }

  return -tM / ecalPh1::Samp_Period;
}

FullSampleVector EcalUncalibRecHitTimingCCAlgo::interpolatePulse(const FullSampleVector& fullpulse, const float t) {
  // t is in ns
  int shift = t / ecalPh1::Samp_Period;
  if (t < 0)
    shift -= 1;
  float timeShift = t - ecalPh1::Samp_Period * shift;
  float tt = timeShift / ecalPh1::Samp_Period;

  FullSampleVector interpPulse;
  // 2nd poly with avg
  unsigned int numberOfSamples = fullpulse.size();
  for (unsigned int i = 1; i < numberOfSamples - 2; ++i) {
    float a = 0.25 * tt * (tt - 1) * fullpulse[i - 1] + (0.25 * (tt - 2) - 0.5 * (tt + 1)) * (tt - 1) * fullpulse[i] +
              (0.25 * (tt + 1) - 0.5 * (tt - 2)) * tt * fullpulse[i + 1] + 0.25 * (tt - 1) * tt * fullpulse[i + 2];
    if (a > 0)
      interpPulse[i] = a;
    else
      interpPulse[i] = 0;
  }
  interpPulse[0] = (0.25 * (tt - 2) - 0.5 * (tt + 1)) * ((tt - 1) * fullpulse[0]) +
                   (0.25 * (tt + 1) + 0.5 * (tt - 2)) * tt * fullpulse[1] + 0.25 * tt * (tt - 1) * fullpulse[2];
  interpPulse[numberOfSamples - 2] = 0.25 * tt * (tt - 1) * fullpulse[numberOfSamples - 3] +
                                     (0.25 * (tt - 2) - 0.5 * (tt + 1)) * (tt - 1) * fullpulse[numberOfSamples - 2] +
                                     (0.25 * (tt + 1) - 0.5 * (tt - 2)) * tt * fullpulse[numberOfSamples - 1];
  interpPulse[numberOfSamples - 1] = 0.5 * tt * (tt - 1) * fullpulse[numberOfSamples - 2] -
                                     (tt * tt - 1) * fullpulse[numberOfSamples - 1] +
                                     (0.25 * (tt + 1) - 0.5 * (tt - 2)) * tt * fullpulse[numberOfSamples - 1];

  FullSampleVector interpPulseShifted;
  for (int i = 0; i < interpPulseShifted.size(); ++i) {
    if (i + shift >= 0 && i + shift < interpPulse.size())
      interpPulseShifted[i] = interpPulse[i + shift];
    else
      interpPulseShifted[i] = 0;
  }
  return interpPulseShifted;
}

float EcalUncalibRecHitTimingCCAlgo::computeCC(const std::vector<double>& samples,
                                               const FullSampleVector& sigmalTemplate,
                                               const float& t) {
  int exclude = 1;
  double powerSamples = 0.;
  for (int i = exclude; i < int(samples.size() - exclude); ++i)
    powerSamples += std::pow(samples[i], 2);

  auto interpolated = interpolatePulse(sigmalTemplate, t);
  double powerTemplate = 0.;
  for (int i = exclude; i < int(interpolated.size() - exclude); ++i)
    powerTemplate += std::pow(interpolated[i], 2);

  double denominator = std::sqrt(powerTemplate * powerSamples);

  double cc = 0.;
  for (int i = exclude; i < int(samples.size() - exclude); ++i) {
    cc += interpolated[i] * samples[i];
  }
  return cc / denominator;
}
