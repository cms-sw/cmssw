#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecChi2Algo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecChi2Algo_HH

/** \class EcalUncalibRecHitRecChi2Algo
  *
  *  Template used to compute the chi2 of an MGPA pulse for in-time and out-of-time signals, algorithm based on the chi2express.  
  *  The in-time chi2 is calculated against the time intercalibrations from the DB while the out-of-time chi2 is calculated
  *  against the Tmax measurement on event by event basis.
  *
  *  \author Konstantinos Theofilatos 02 Feb 2010 
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

template <class C>
class EcalUncalibRecHitRecChi2Algo {
public:
  // destructor
  virtual ~EcalUncalibRecHitRecChi2Algo(){};

  EcalUncalibRecHitRecChi2Algo(){};
  EcalUncalibRecHitRecChi2Algo(const C& dataFrame,
                               const double amplitude,
                               const EcalTimeCalibConstant& timeIC,
                               const double amplitudeOutOfTime,
                               const double jitter,
                               const double* pedestals,
                               const double* pedestalsRMS,
                               const double* gainRatios,
                               const EcalShapeBase& testbeamPulseShape,
                               const std::vector<double>& chi2Parameters);

  virtual double chi2() { return chi2_; }
  virtual double chi2OutOfTime() { return chi2OutOfTime_; }

private:
  double chi2_;
  double chi2OutOfTime_;
};

template <class C>
EcalUncalibRecHitRecChi2Algo<C>::EcalUncalibRecHitRecChi2Algo(const C& dataFrame,
                                                              const double amplitude,
                                                              const EcalTimeCalibConstant& timeIC,
                                                              const double amplitudeOutOfTime,
                                                              const double jitter,
                                                              const double* pedestals,
                                                              const double* pedestalsRMS,
                                                              const double* gainRatios,
                                                              const EcalShapeBase& testbeamPulseShape,
                                                              const std::vector<double>& chi2Parameters) {
  double noise_A = chi2Parameters[0];  // noise term for in-time chi2
  double const_A = chi2Parameters[1];  // constant term for in-time chi2
  double noise_B = chi2Parameters[2];  // noise term for out-of-time chi2
  double const_B = chi2Parameters[3];  // constant term for out-of-time chi2

  chi2_ = 0;
  chi2OutOfTime_ = 0;
  double S_0 = 0;      // will store the first mgpa sample
  double ped_ave = 0;  // will store the average pedestal

  int gainId0 = 1;
  int iGainSwitch = 0;
  bool isSaturated = false;
  for (int iSample = 0; iSample < C::MAXSAMPLES;
       iSample++)  // if gain switch use later the pedestal RMS, otherwise we use the pedestal from the DB
  {
    int gainId = dataFrame.sample(iSample).gainId();
    if (gainId == 0) {
      gainId = 3;  // if saturated, treat it as G1
      isSaturated = true;
    }
    if (gainId != gainId0)
      iGainSwitch = 1;

    if (gainId == 1 && iSample == 0)
      S_0 = dataFrame.sample(iSample).adc();  // take only first presample to estimate the pedestal
    if (gainId == 1 && iSample < 3)
      ped_ave += (1 / 3.0) * dataFrame.sample(iSample).adc();  // take first 3 presamples to estimate the pedestal
  }

  // compute testbeamPulseShape shape parameters
  double ADC_clock = 25;  // 25 ns
  double risingTime = testbeamPulseShape.timeToRise();
  double tzero = risingTime - 5 * ADC_clock;  // 5 samples before the peak

  double shiftTime = +timeIC;                       // we put positive here
  double shiftTimeOutOfTime = -jitter * ADC_clock;  // we put negative here

  bool readoutError = false;

  for (int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
    int gainId = dataFrame.sample(iSample).gainId();
    if (dataFrame.sample(iSample).adc() == 0)
      readoutError = true;
    if (gainId == 0)
      continue;  // skip saturated samples

    double ped =
        !iGainSwitch ? ped_ave : pedestals[gainId - 1];  // use dynamic pedestal for G12 and average pedestal for G6,G1
                                                         //double pedRMS = pedestalsRMS[gainId-1];
    double S_i = double(dataFrame.sample(iSample).adc());

    // --- calculate in-time chi2

    double f_i = (testbeamPulseShape)(tzero + shiftTime + iSample * ADC_clock);
    double R_i = (S_i - ped) * gainRatios[gainId - 1] - f_i * amplitude;
    double R_iErrorSquare = noise_A * noise_A + const_A * const_A * amplitude * amplitude;

    chi2_ += R_i * R_i / R_iErrorSquare;

    // --- calculate out-of-time chi2

    double g_i = (testbeamPulseShape)(tzero + shiftTimeOutOfTime + iSample * ADC_clock);  // calculate out of time chi2

    double R_iOutOfTime = (S_i - S_0) * gainRatios[gainId - 1] - g_i * amplitudeOutOfTime;
    double R_iOutOfTimeErrorSquare = noise_B * noise_B + const_B * const_B * amplitudeOutOfTime * amplitudeOutOfTime;

    chi2OutOfTime_ += R_iOutOfTime * R_iOutOfTime / R_iOutOfTimeErrorSquare;
  }

  if (!isSaturated && !iGainSwitch && chi2_ > 0 && chi2OutOfTime_ > 0) {
    chi2_ = 7 * (3 + log(chi2_));
    chi2_ = chi2_ < 0 ? 0 : chi2_;  // this is just a convinient mapping for storing in the calibRecHit bit map
    chi2OutOfTime_ = 7 * (3 + log(chi2OutOfTime_));
    chi2OutOfTime_ = chi2OutOfTime_ < 0 ? 0 : chi2OutOfTime_;
  } else {
    chi2_ = 0;
    chi2OutOfTime_ = 0;
  }

  if (readoutError)  // rare situation
  {
    chi2_ = 99.0;  // chi2 is very large in these cases, put a code value to discriminate against normal noise
    chi2OutOfTime_ = 99.0;
  }
}

#endif
