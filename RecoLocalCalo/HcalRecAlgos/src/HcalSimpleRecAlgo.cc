#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSimpleRecAlgo.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include "RecoLocalCalo/HcalRecAlgos/src/HcalTDCReco.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/rawEnergy.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCorrectionFunctions.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <algorithm>
#include <cmath>

//--- temporary for printouts
// #include<iostream>

constexpr double MaximumFractionalError = 0.002;  // 0.2% error allowed from this source

HcalSimpleRecAlgo::HcalSimpleRecAlgo(bool correctForTimeslew, bool correctForPulse, float phaseNS)
    : correctForTimeslew_(correctForTimeslew),
      correctForPulse_(correctForPulse),
      phaseNS_(phaseNS),
      runnum_(0),
      setLeakCorrection_(false),
      puCorrMethod_(0) {
  hcalTimeSlew_delay_ = nullptr;
  pulseCorr_ = std::make_unique<HcalPulseContainmentManager>(MaximumFractionalError);
}

void HcalSimpleRecAlgo::beginRun(edm::EventSetup const& es) {
  edm::ESHandle<HcalTimeSlew> delay;
  es.get<HcalTimeSlewRecord>().get("HBHE", delay);
  hcalTimeSlew_delay_ = &*delay;

  pulseCorr_->beginRun(es);
}

void HcalSimpleRecAlgo::endRun() {}

void HcalSimpleRecAlgo::initPulseCorr(int toadd) {}

void HcalSimpleRecAlgo::setRecoParams(
    bool correctForTimeslew, bool correctForPulse, bool setLeakCorrection, int pileupCleaningID, float phaseNS) {
  correctForTimeslew_ = correctForTimeslew;
  correctForPulse_ = correctForPulse;
  phaseNS_ = phaseNS;
  setLeakCorrection_ = setLeakCorrection;
  pileupCleaningID_ = pileupCleaningID;
}

void HcalSimpleRecAlgo::setLeakCorrection() { setLeakCorrection_ = true; }

void HcalSimpleRecAlgo::setHFPileupCorrection(std::shared_ptr<AbsOOTPileupCorrection> corr) { hfPileupCorr_ = corr; }

void HcalSimpleRecAlgo::setHOPileupCorrection(std::shared_ptr<AbsOOTPileupCorrection> corr) { hoPileupCorr_ = corr; }

void HcalSimpleRecAlgo::setBXInfo(const BunchXParameter* info, const unsigned lenInfo) {
  bunchCrossingInfo_ = info;
  lenBunchCrossingInfo_ = lenInfo;
}

///Timeshift correction for the HF PMTs.
static float timeshift_ns_hf(float wpksamp);

/// Leak correction
static float leakCorr(double energy);

namespace HcalSimpleRecAlgoImpl {
  template <class Digi>
  inline float recoHFTime(
      const Digi& digi, const int maxI, const double amp_fC, const bool slewCorrect, double maxA, float t0, float t2) {
    // Handle negative excursions by moving "zero":
    float zerocorr = std::min(t0, t2);
    if (zerocorr < 0.f) {
      t0 -= zerocorr;
      t2 -= zerocorr;
      maxA -= zerocorr;
    }

    // pair the peak with the larger of the two neighboring time samples
    float wpksamp = 0.f;
    if (t0 > t2) {
      wpksamp = t0 + maxA;
      if (wpksamp != 0.f)
        wpksamp = maxA / wpksamp;
    } else {
      wpksamp = maxA + t2;
      if (wpksamp != 0.f)
        wpksamp = 1. + (t2 / wpksamp);
    }

    float time = (maxI - digi.presamples()) * 25.0 + timeshift_ns_hf(wpksamp);

    if (slewCorrect && amp_fC > 0.0) {
      // -5.12327 - put in calibs.timecorr()
      double tslew = exp(0.337681 - 5.94689e-4 * amp_fC) + exp(2.44628 - 1.34888e-2 * amp_fC);
      time -= (float)tslew;
    }

    return time;
  }

  template <class Digi>
  inline void removePileup(const Digi& digi,
                           const HcalCoder& coder,
                           const HcalCalibrations& calibs,
                           const int ifirst,
                           const int n,
                           const bool pulseCorrect,
                           const HcalPulseContainmentCorrection* corr,
                           const AbsOOTPileupCorrection* pileupCorrection,
                           const BunchXParameter* bxInfo,
                           const unsigned lenInfo,
                           double* p_maxA,
                           double* p_ampl,
                           double* p_uncorr_ampl,
                           double* p_fc_ampl,
                           int* p_nRead,
                           int* p_maxI,
                           bool* leakCorrApplied,
                           float* p_t0,
                           float* p_t2) {
    CaloSamples cs;
    coder.adc2fC(digi, cs);
    const int nRead = cs.size();
    const int iStop = std::min(nRead, n + ifirst);

    // Signal energy will be calculated both with
    // and without OOT pileup corrections. Try to
    // arrange the calculations so that we do not
    // repeat them.
    double uncorrectedEnergy[CaloSamples::MAXSAMPLES]{}, buf[CaloSamples::MAXSAMPLES]{};
    double* correctedEnergy = nullptr;
    double fc_ampl = 0.0, corr_fc_ampl = 0.0;
    bool pulseShapeCorrApplied = false, readjustTiming = false;
    *leakCorrApplied = false;

    if (pileupCorrection) {
      correctedEnergy = &buf[0];

      double correctionInput[CaloSamples::MAXSAMPLES]{};
      double gains[CaloSamples::MAXSAMPLES]{};

      for (int i = 0; i < nRead; ++i) {
        const int capid = digi[i].capid();
        correctionInput[i] = cs[i] - calibs.pedestal(capid);
        gains[i] = calibs.respcorrgain(capid);
      }

      for (int i = ifirst; i < iStop; ++i)
        fc_ampl += correctionInput[i];

      const bool useGain = pileupCorrection->inputIsEnergy();
      for (int i = 0; i < nRead; ++i) {
        uncorrectedEnergy[i] = correctionInput[i] * gains[i];
        if (useGain)
          correctionInput[i] = uncorrectedEnergy[i];
      }

      pileupCorrection->apply(digi.id(),
                              correctionInput,
                              nRead,
                              bxInfo,
                              lenInfo,
                              ifirst,
                              n,
                              correctedEnergy,
                              CaloSamples::MAXSAMPLES,
                              &pulseShapeCorrApplied,
                              leakCorrApplied,
                              &readjustTiming);
      if (useGain) {
        // Gain factors have been already applied.
        // Divide by them for accumulating corr_fc_ampl.
        for (int i = ifirst; i < iStop; ++i)
          if (gains[i])
            corr_fc_ampl += correctedEnergy[i] / gains[i];
      } else {
        for (int i = ifirst; i < iStop; ++i)
          corr_fc_ampl += correctedEnergy[i];
        for (int i = 0; i < nRead; ++i)
          correctedEnergy[i] *= gains[i];
      }
    } else {
      correctedEnergy = &uncorrectedEnergy[0];

      // In this situation, we do not need to process all time slices
      const int istart = std::max(ifirst - 1, 0);
      const int iend = std::min(n + ifirst + 1, nRead);
      for (int i = istart; i < iend; ++i) {
        const int capid = digi[i].capid();
        float ta = cs[i] - calibs.pedestal(capid);
        if (i >= ifirst && i < iStop)
          fc_ampl += ta;
        ta *= calibs.respcorrgain(capid);
        uncorrectedEnergy[i] = ta;
      }
      corr_fc_ampl = fc_ampl;
    }

    // Uncorrected and corrected energies
    double ampl = 0.0, corr_ampl = 0.0;
    for (int i = ifirst; i < iStop; ++i) {
      ampl += uncorrectedEnergy[i];
      corr_ampl += correctedEnergy[i];
    }

    // Apply phase-based amplitude correction:
    if (corr && pulseCorrect) {
      ampl *= corr->getCorrection(fc_ampl);
      if (pileupCorrection) {
        if (!pulseShapeCorrApplied)
          corr_ampl *= corr->getCorrection(corr_fc_ampl);
      } else
        corr_ampl = ampl;
    }

    // Which energies we want to use for timing?
    const double* etime = readjustTiming ? &correctedEnergy[0] : &uncorrectedEnergy[0];
    int maxI = -1;
    double maxA = -1.e300;
    for (int i = ifirst; i < iStop; ++i)
      if (etime[i] > maxA) {
        maxA = etime[i];
        maxI = i;
      }

    // Fill out the output
    *p_maxA = maxA;
    *p_ampl = corr_ampl;
    *p_uncorr_ampl = ampl;
    *p_fc_ampl = readjustTiming ? corr_fc_ampl : fc_ampl;
    *p_nRead = nRead;
    *p_maxI = maxI;

    if (maxI <= 0 || maxI >= (nRead - 1)) {
      LogDebug("HCAL Pulse") << "HcalSimpleRecAlgoImpl::removePileup :"
                             << " Invalid max amplitude position, "
                             << " max Amplitude: " << maxI << " first: " << ifirst << " last: " << ifirst + n
                             << std::endl;
      *p_t0 = 0.f;
      *p_t2 = 0.f;
    } else {
      *p_t0 = etime[maxI - 1];
      *p_t2 = etime[maxI + 1];
    }
  }

  template <class Digi, class RecHit>
  inline RecHit reco(const Digi& digi,
                     const HcalCoder& coder,
                     const HcalCalibrations& calibs,
                     const int ifirst,
                     const int n,
                     const bool slewCorrect,
                     const bool pulseCorrect,
                     const HcalPulseContainmentCorrection* corr,
                     const HcalTimeSlew::BiasSetting slewFlavor,
                     const int runnum,
                     const bool useLeak,
                     const AbsOOTPileupCorrection* pileupCorrection,
                     const BunchXParameter* bxInfo,
                     const unsigned lenInfo,
                     const int puCorrMethod,
                     const HcalTimeSlew* hcalTimeSlew_delay_) {
    double fc_ampl = 0, ampl = 0, uncorr_ampl = 0, m3_ampl = 0, maxA = -1.e300;
    int nRead = 0, maxI = -1;
    bool leakCorrApplied = false;
    float t0 = 0, t2 = 0;
    float time = -9999;

    // Disable method 1 inside the removePileup function this way!
    // Some code in removePileup does NOT do pileup correction & to make sure maximum share of code
    const AbsOOTPileupCorrection* inputAbsOOTpuCorr = (puCorrMethod == 1 ? pileupCorrection : nullptr);

    removePileup(digi,
                 coder,
                 calibs,
                 ifirst,
                 n,
                 pulseCorrect,
                 corr,
                 inputAbsOOTpuCorr,
                 bxInfo,
                 lenInfo,
                 &maxA,
                 &ampl,
                 &uncorr_ampl,
                 &fc_ampl,
                 &nRead,
                 &maxI,
                 &leakCorrApplied,
                 &t0,
                 &t2);

    if (maxI > 0 && maxI < (nRead - 1)) {
      // Handle negative excursions by moving "zero":
      float minA = t0;
      if (maxA < minA)
        minA = maxA;
      if (t2 < minA)
        minA = t2;
      if (minA < 0) {
        maxA -= minA;
        t0 -= minA;
        t2 -= minA;
      }  // positivizes all samples

      float wpksamp = (t0 + maxA + t2);
      if (wpksamp != 0)
        wpksamp = (maxA + 2.0 * t2) / wpksamp;
      time = (maxI - digi.presamples()) * 25.0 + timeshift_ns_hbheho(wpksamp);

      if (slewCorrect)
        time -= hcalTimeSlew_delay_->delay(std::max(1.0, fc_ampl), slewFlavor);

      time = time - calibs.timecorr();  // time calibration
    }

    // Correction for a leak to pre-sample
    if (useLeak && !leakCorrApplied) {
      uncorr_ampl *= leakCorr(uncorr_ampl);
      if (puCorrMethod < 2)
        ampl *= leakCorr(ampl);
    }

    RecHit rh(digi.id(), ampl, time);
    setRawEnergy(rh, static_cast<float>(uncorr_ampl));
    setAuxEnergy(rh, static_cast<float>(m3_ampl));
    return rh;
  }
}  // namespace HcalSimpleRecAlgoImpl

HORecHit HcalSimpleRecAlgo::reconstruct(
    const HODataFrame& digi, int first, int toadd, const HcalCoder& coder, const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HODataFrame, HORecHit>(digi,
                                                            coder,
                                                            calibs,
                                                            first,
                                                            toadd,
                                                            correctForTimeslew_,
                                                            correctForPulse_,
                                                            pulseCorr_->get(digi.id(), toadd, phaseNS_),
                                                            HcalTimeSlew::Slow,
                                                            runnum_,
                                                            false,
                                                            hoPileupCorr_.get(),
                                                            bunchCrossingInfo_,
                                                            lenBunchCrossingInfo_,
                                                            puCorrMethod_,
                                                            hcalTimeSlew_delay_);
}

HcalCalibRecHit HcalSimpleRecAlgo::reconstruct(const HcalCalibDataFrame& digi,
                                               int first,
                                               int toadd,
                                               const HcalCoder& coder,
                                               const HcalCalibrations& calibs) const {
  return HcalSimpleRecAlgoImpl::reco<HcalCalibDataFrame, HcalCalibRecHit>(digi,
                                                                          coder,
                                                                          calibs,
                                                                          first,
                                                                          toadd,
                                                                          correctForTimeslew_,
                                                                          correctForPulse_,
                                                                          pulseCorr_->get(digi.id(), toadd, phaseNS_),
                                                                          HcalTimeSlew::Fast,
                                                                          runnum_,
                                                                          false,
                                                                          nullptr,
                                                                          bunchCrossingInfo_,
                                                                          lenBunchCrossingInfo_,
                                                                          puCorrMethod_,
                                                                          hcalTimeSlew_delay_);
}

HFRecHit HcalSimpleRecAlgo::reconstruct(const HFDataFrame& digi,
                                        const int first,
                                        const int toadd,
                                        const HcalCoder& coder,
                                        const HcalCalibrations& calibs) const {
  const HcalPulseContainmentCorrection* corr = pulseCorr_->get(digi.id(), toadd, phaseNS_);

  double amp_fC, ampl, uncorr_ampl, maxA;
  int nRead, maxI;
  bool leakCorrApplied;
  float t0, t2;

  HcalSimpleRecAlgoImpl::removePileup(digi,
                                      coder,
                                      calibs,
                                      first,
                                      toadd,
                                      correctForPulse_,
                                      corr,
                                      hfPileupCorr_.get(),
                                      bunchCrossingInfo_,
                                      lenBunchCrossingInfo_,
                                      &maxA,
                                      &ampl,
                                      &uncorr_ampl,
                                      &amp_fC,
                                      &nRead,
                                      &maxI,
                                      &leakCorrApplied,
                                      &t0,
                                      &t2);

  float time = -9999.f;
  if (maxI > 0 && maxI < (nRead - 1))
    time = HcalSimpleRecAlgoImpl::recoHFTime(digi, maxI, amp_fC, correctForTimeslew_, maxA, t0, t2) - calibs.timecorr();

  HFRecHit rh(digi.id(), ampl, time);
  setRawEnergy(rh, static_cast<float>(uncorr_ampl));
  return rh;
}

HFRecHit HcalSimpleRecAlgo::reconstructQIE10(const QIE10DataFrame& digi,
                                             const int first,
                                             const int toadd,
                                             const HcalCoder& coder,
                                             const HcalCalibrations& calibs) const {
  const HcalPulseContainmentCorrection* corr = pulseCorr_->get(digi.id(), toadd, phaseNS_);

  double amp_fC, ampl, uncorr_ampl, maxA;
  int nRead, maxI;
  bool leakCorrApplied;
  float t0, t2;

  HcalSimpleRecAlgoImpl::removePileup(digi,
                                      coder,
                                      calibs,
                                      first,
                                      toadd,
                                      correctForPulse_,
                                      corr,
                                      hfPileupCorr_.get(),
                                      bunchCrossingInfo_,
                                      lenBunchCrossingInfo_,
                                      &maxA,
                                      &ampl,
                                      &uncorr_ampl,
                                      &amp_fC,
                                      &nRead,
                                      &maxI,
                                      &leakCorrApplied,
                                      &t0,
                                      &t2);

  float time = -9999.f;
  if (maxI > 0 && maxI < (nRead - 1))
    time = HcalSimpleRecAlgoImpl::recoHFTime(digi, maxI, amp_fC, correctForTimeslew_, maxA, t0, t2) - calibs.timecorr();

  HFRecHit rh(digi.id(), ampl, time);
  setRawEnergy(rh, static_cast<float>(uncorr_ampl));
  return rh;
}

/// Ugly hack to apply energy corrections to some HB- cells
float hbminus_special_ecorr(int ieta, int iphi, double energy, int runnum) {
  // return energy correction factor for HBM channels
  // iphi=6 ieta=(-1,-15) and iphi=32 ieta=(-1,-7)
  // I.Vodopianov 28 Feb. 2011
  static const float low32[7] = {0.741, 0.721, 0.730, 0.698, 0.708, 0.751, 0.861};
  static const float high32[7] = {0.973, 0.925, 0.900, 0.897, 0.950, 0.935, 1};
  static const float low6[15] = {
      0.635, 0.623, 0.670, 0.633, 0.644, 0.648, 0.600, 0.570, 0.595, 0.554, 0.505, 0.513, 0.515, 0.561, 0.579};
  static const float high6[15] = {
      0.875, 0.937, 0.942, 0.900, 0.922, 0.925, 0.901, 0.850, 0.852, 0.818, 0.731, 0.717, 0.782, 0.853, 0.778};

  double slope, mid, en;
  double corr = 1.0;

  if (!(iphi == 6 && ieta < 0 && ieta > -16) && !(iphi == 32 && ieta < 0 && ieta > -8))
    return corr;

  int jeta = -ieta - 1;
  double xeta = (double)ieta;
  if (energy > 0.)
    en = energy;
  else
    en = 0.;

  if (iphi == 32) {
    slope = 0.2272;
    mid = 17.14 + 0.7147 * xeta;
    if (en > 100.)
      corr = high32[jeta];
    else
      corr = low32[jeta] + (high32[jeta] - low32[jeta]) / (1.0 + exp(-(en - mid) * slope));
  } else if (iphi == 6 && runnum < 216091) {
    slope = 0.1956;
    mid = 15.96 + 0.3075 * xeta;
    if (en > 100.0)
      corr = high6[jeta];
    else
      corr = low6[jeta] + (high6[jeta] - low6[jeta]) / (1.0 + exp(-(en - mid) * slope));
  }

  //  std::cout << "HBHE cell:  ieta, iphi = " << ieta << "  " << iphi
  //	    << "  ->  energy = " << en << "   corr = " << corr << std::endl;

  return corr;
}

// Actual leakage (to pre-sample) correction
float leakCorr(double energy) {
  double corr = 1.0;
  return corr;
}

// timeshift implementation

static const float wpksamp0_hbheho = 0.5;
static const int num_bins_hbheho = 61;

static const float actual_ns_hbheho[num_bins_hbheho] = {
    -5.44000,  // 0.500, 0.000-0.017
    -4.84250,  // 0.517, 0.017-0.033
    -4.26500,  // 0.533, 0.033-0.050
    -3.71000,  // 0.550, 0.050-0.067
    -3.18000,  // 0.567, 0.067-0.083
    -2.66250,  // 0.583, 0.083-0.100
    -2.17250,  // 0.600, 0.100-0.117
    -1.69000,  // 0.617, 0.117-0.133
    -1.23000,  // 0.633, 0.133-0.150
    -0.78000,  // 0.650, 0.150-0.167
    -0.34250,  // 0.667, 0.167-0.183
    0.08250,   // 0.683, 0.183-0.200
    0.50250,   // 0.700, 0.200-0.217
    0.90500,   // 0.717, 0.217-0.233
    1.30500,   // 0.733, 0.233-0.250
    1.69500,   // 0.750, 0.250-0.267
    2.07750,   // 0.767, 0.267-0.283
    2.45750,   // 0.783, 0.283-0.300
    2.82500,   // 0.800, 0.300-0.317
    3.19250,   // 0.817, 0.317-0.333
    3.55750,   // 0.833, 0.333-0.350
    3.91750,   // 0.850, 0.350-0.367
    4.27500,   // 0.867, 0.367-0.383
    4.63000,   // 0.883, 0.383-0.400
    4.98500,   // 0.900, 0.400-0.417
    5.33750,   // 0.917, 0.417-0.433
    5.69500,   // 0.933, 0.433-0.450
    6.05000,   // 0.950, 0.450-0.467
    6.40500,   // 0.967, 0.467-0.483
    6.77000,   // 0.983, 0.483-0.500
    7.13500,   // 1.000, 0.500-0.517
    7.50000,   // 1.017, 0.517-0.533
    7.88250,   // 1.033, 0.533-0.550
    8.26500,   // 1.050, 0.550-0.567
    8.66000,   // 1.067, 0.567-0.583
    9.07000,   // 1.083, 0.583-0.600
    9.48250,   // 1.100, 0.600-0.617
    9.92750,   // 1.117, 0.617-0.633
    10.37750,  // 1.133, 0.633-0.650
    10.87500,  // 1.150, 0.650-0.667
    11.38000,  // 1.167, 0.667-0.683
    11.95250,  // 1.183, 0.683-0.700
    12.55000,  // 1.200, 0.700-0.717
    13.22750,  // 1.217, 0.717-0.733
    13.98500,  // 1.233, 0.733-0.750
    14.81500,  // 1.250, 0.750-0.767
    15.71500,  // 1.267, 0.767-0.783
    16.63750,  // 1.283, 0.783-0.800
    17.53750,  // 1.300, 0.800-0.817
    18.38500,  // 1.317, 0.817-0.833
    19.16500,  // 1.333, 0.833-0.850
    19.89750,  // 1.350, 0.850-0.867
    20.59250,  // 1.367, 0.867-0.883
    21.24250,  // 1.383, 0.883-0.900
    21.85250,  // 1.400, 0.900-0.917
    22.44500,  // 1.417, 0.917-0.933
    22.99500,  // 1.433, 0.933-0.950
    23.53250,  // 1.450, 0.950-0.967
    24.03750,  // 1.467, 0.967-0.983
    24.53250,  // 1.483, 0.983-1.000
    25.00000   // 1.500, 1.000-1.017 - keep for interpolation
};

float timeshift_ns_hbheho(float wpksamp) {
  float flx = (num_bins_hbheho - 1) * (wpksamp - wpksamp0_hbheho);
  int index = (int)flx;
  float yval;

  if (index < 0)
    return actual_ns_hbheho[0];
  else if (index >= num_bins_hbheho - 1)
    return actual_ns_hbheho[num_bins_hbheho - 1];

  // else interpolate:
  float y1 = actual_ns_hbheho[index];
  float y2 = actual_ns_hbheho[index + 1];

  yval = y1 + (y2 - y1) * (flx - (float)index);

  return yval;
}

static const int num_bins_hf = 101;
static const float wpksamp0_hf = 0.5;

static const float actual_ns_hf[num_bins_hf] = {
    0.00250,   // 0.000-0.010
    0.04500,   // 0.010-0.020
    0.08750,   // 0.020-0.030
    0.13000,   // 0.030-0.040
    0.17250,   // 0.040-0.050
    0.21500,   // 0.050-0.060
    0.26000,   // 0.060-0.070
    0.30250,   // 0.070-0.080
    0.34500,   // 0.080-0.090
    0.38750,   // 0.090-0.100
    0.42750,   // 0.100-0.110
    0.46000,   // 0.110-0.120
    0.49250,   // 0.120-0.130
    0.52500,   // 0.130-0.140
    0.55750,   // 0.140-0.150
    0.59000,   // 0.150-0.160
    0.62250,   // 0.160-0.170
    0.65500,   // 0.170-0.180
    0.68750,   // 0.180-0.190
    0.72000,   // 0.190-0.200
    0.75250,   // 0.200-0.210
    0.78500,   // 0.210-0.220
    0.81750,   // 0.220-0.230
    0.85000,   // 0.230-0.240
    0.88250,   // 0.240-0.250
    0.91500,   // 0.250-0.260
    0.95500,   // 0.260-0.270
    0.99250,   // 0.270-0.280
    1.03250,   // 0.280-0.290
    1.07000,   // 0.290-0.300
    1.10750,   // 0.300-0.310
    1.14750,   // 0.310-0.320
    1.18500,   // 0.320-0.330
    1.22500,   // 0.330-0.340
    1.26250,   // 0.340-0.350
    1.30000,   // 0.350-0.360
    1.34000,   // 0.360-0.370
    1.37750,   // 0.370-0.380
    1.41750,   // 0.380-0.390
    1.48750,   // 0.390-0.400
    1.55750,   // 0.400-0.410
    1.62750,   // 0.410-0.420
    1.69750,   // 0.420-0.430
    1.76750,   // 0.430-0.440
    1.83750,   // 0.440-0.450
    1.90750,   // 0.450-0.460
    2.06750,   // 0.460-0.470
    2.23250,   // 0.470-0.480
    2.40000,   // 0.480-0.490
    2.82250,   // 0.490-0.500
    3.81000,   // 0.500-0.510
    6.90500,   // 0.510-0.520
    8.99250,   // 0.520-0.530
    10.50000,  // 0.530-0.540
    11.68250,  // 0.540-0.550
    12.66250,  // 0.550-0.560
    13.50250,  // 0.560-0.570
    14.23750,  // 0.570-0.580
    14.89750,  // 0.580-0.590
    15.49000,  // 0.590-0.600
    16.03250,  // 0.600-0.610
    16.53250,  // 0.610-0.620
    17.00000,  // 0.620-0.630
    17.44000,  // 0.630-0.640
    17.85250,  // 0.640-0.650
    18.24000,  // 0.650-0.660
    18.61000,  // 0.660-0.670
    18.96750,  // 0.670-0.680
    19.30500,  // 0.680-0.690
    19.63000,  // 0.690-0.700
    19.94500,  // 0.700-0.710
    20.24500,  // 0.710-0.720
    20.54000,  // 0.720-0.730
    20.82250,  // 0.730-0.740
    21.09750,  // 0.740-0.750
    21.37000,  // 0.750-0.760
    21.62750,  // 0.760-0.770
    21.88500,  // 0.770-0.780
    22.13000,  // 0.780-0.790
    22.37250,  // 0.790-0.800
    22.60250,  // 0.800-0.810
    22.83000,  // 0.810-0.820
    23.04250,  // 0.820-0.830
    23.24500,  // 0.830-0.840
    23.44250,  // 0.840-0.850
    23.61000,  // 0.850-0.860
    23.77750,  // 0.860-0.870
    23.93500,  // 0.870-0.880
    24.05500,  // 0.880-0.890
    24.17250,  // 0.890-0.900
    24.29000,  // 0.900-0.910
    24.40750,  // 0.910-0.920
    24.48250,  // 0.920-0.930
    24.55500,  // 0.930-0.940
    24.62500,  // 0.940-0.950
    24.69750,  // 0.950-0.960
    24.77000,  // 0.960-0.970
    24.84000,  // 0.970-0.980
    24.91250,  // 0.980-0.990
    24.95500,  // 0.990-1.000
    24.99750,  // 1.000-1.010 - keep for interpolation
};

float timeshift_ns_hf(float wpksamp) {
  float flx = (num_bins_hf - 1) * (wpksamp - wpksamp0_hf);
  int index = (int)flx;
  float yval;

  if (index < 0)
    return actual_ns_hf[0];
  else if (index >= num_bins_hf - 1)
    return actual_ns_hf[num_bins_hf - 1];

  // else interpolate:
  float y1 = actual_ns_hf[index];
  float y2 = actual_ns_hf[index + 1];

  // float delta_x  = 1/(float)num_bins_hf;
  // yval = y1 + (y2-y1)*(flx-(float)index)/delta_x;

  yval = y1 + (y2 - y1) * (flx - (float)index);
  return yval;
}
