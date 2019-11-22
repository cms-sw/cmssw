#include <algorithm>

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/SimpleHBHEPhase1Algo.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCorrectionFunctions.h"

#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHitAuxSetter.h"
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Maximum fractional error for calculating Method 0
// pulse containment correction
constexpr float PulseContainmentFractionalError = 0.002f;

SimpleHBHEPhase1Algo::SimpleHBHEPhase1Algo(const int firstSampleShift,
                                           const int samplesToAdd,
                                           const float phaseNS,
                                           const float timeShift,
                                           const bool correctForPhaseContainment,
                                           const bool applyLegacyHBMCorrection,
                                           std::unique_ptr<PulseShapeFitOOTPileupCorrection> m2,
                                           std::unique_ptr<HcalDeterministicFit> detFit,
                                           std::unique_ptr<MahiFit> mahi)
    : pulseCorr_(PulseContainmentFractionalError),
      firstSampleShift_(firstSampleShift),
      samplesToAdd_(samplesToAdd),
      phaseNS_(phaseNS),
      timeShift_(timeShift),
      runnum_(0),
      corrFPC_(correctForPhaseContainment),
      applyLegacyHBMCorrection_(applyLegacyHBMCorrection),
      psFitOOTpuCorr_(std::move(m2)),
      hltOOTpuCorr_(std::move(detFit)),
      mahiOOTpuCorr_(std::move(mahi)) {
  hcalTimeSlew_delay_ = nullptr;
}

void SimpleHBHEPhase1Algo::beginRun(const edm::Run& r, const edm::EventSetup& es) {
  edm::ESHandle<HcalTimeSlew> delay;
  es.get<HcalTimeSlewRecord>().get("HBHE", delay);
  hcalTimeSlew_delay_ = &*delay;

  runnum_ = r.run();
  pulseCorr_.beginRun(es);
}

void SimpleHBHEPhase1Algo::endRun() { runnum_ = 0; }

HBHERecHit SimpleHBHEPhase1Algo::reconstruct(const HBHEChannelInfo& info,
                                             const HcalRecoParam* params,
                                             const HcalCalibrations& calibs,
                                             const bool isData) {
  const HcalDetId channelId(info.id());

  // Calculate "Method 0" quantities
  float m0t = 0.f, m0E = 0.f;
  {
    int ibeg = static_cast<int>(info.soi()) + firstSampleShift_;
    if (ibeg < 0)
      ibeg = 0;
    const int nSamplesToAdd = params ? params->samplesToAdd() : samplesToAdd_;
    const double fc_ampl = info.chargeInWindow(ibeg, ibeg + nSamplesToAdd);
    const bool applyContainment = params ? params->correctForPhaseContainment() : corrFPC_;
    const float phasens = params ? params->correctionPhaseNS() : phaseNS_;
    m0E = m0Energy(info, fc_ampl, applyContainment, phasens, nSamplesToAdd);
    m0E *= hbminusCorrectionFactor(channelId, m0E, isData);
    m0t = m0Time(info, fc_ampl, nSamplesToAdd);
  }

  // Run "Method 2"
  float m2t = 0.f, m2E = 0.f, chi2 = -1.f;
  bool useTriple = false;
  const PulseShapeFitOOTPileupCorrection* method2 = psFitOOTpuCorr_.get();
  if (method2) {
    psFitOOTpuCorr_->setPulseShapeTemplate(
        theHcalPulseShapes_.getShape(info.recoShape()), !info.hasTimeInfo(), info.nSamples(), hcalTimeSlew_delay_);
    // "phase1Apply" call below sets m2E, m2t, useTriple, and chi2.
    // These parameters are pased by non-const reference.
    method2->phase1Apply(info, m2E, m2t, useTriple, chi2);
    m2E *= hbminusCorrectionFactor(channelId, m2E, isData);
  }

  // Run "Method 3"
  float m3t = 0.f, m3E = 0.f;
  const HcalDeterministicFit* method3 = hltOOTpuCorr_.get();
  if (method3) {
    // "phase1Apply" sets m3E and m3t (pased by non-const reference)
    method3->phase1Apply(info, m3E, m3t, hcalTimeSlew_delay_);
    m3E *= hbminusCorrectionFactor(channelId, m3E, isData);
  }

  // Run Mahi
  float m4E = 0.f, m4chi2 = -1.f;
  float m4T = 0.f;
  bool m4UseTriple = false;

  const MahiFit* mahi = mahiOOTpuCorr_.get();

  if (mahi) {
    mahiOOTpuCorr_->setPulseShapeTemplate(
        theHcalPulseShapes_.getShape(info.recoShape()), info.hasTimeInfo(), hcalTimeSlew_delay_, info.nSamples());
    mahi->phase1Apply(info, m4E, m4T, m4UseTriple, m4chi2);
    m4E *= hbminusCorrectionFactor(channelId, m4E, isData);
  }

  // Finally, construct the rechit
  HBHERecHit rh;

  float rhE = m0E;
  float rht = m0t;
  float rhX = -1.f;
  if (mahi) {
    rhE = m4E;
    rht = m4T;
    rhX = m4chi2;
  } else if (method2) {
    rhE = m2E;
    rht = m2t;
    rhX = chi2;
  } else if (method3) {
    rhE = m3E;
    rht = m3t;
  }
  float tdcTime = info.soiRiseTime();
  if (!HcalSpecialTimes::isSpecial(tdcTime))
    tdcTime += timeShift_;
  rh = HBHERecHit(channelId, rhE, rht, tdcTime);
  rh.setRawEnergy(m0E);
  rh.setAuxEnergy(m3E);
  rh.setChiSquared(rhX);

  // Set rechit aux words
  HBHERecHitAuxSetter::setAux(info, &rh);

  // Set some rechit flags (here, for Method 2/Mahi)
  if (useTriple || m4UseTriple)
    rh.setFlagField(1, HcalPhase1FlagLabels::HBHEPulseFitBit);

  return rh;
}

float SimpleHBHEPhase1Algo::hbminusCorrectionFactor(const HcalDetId& cell,
                                                    const float energy,
                                                    const bool isRealData) const {
  float corr = 1.f;
  if (applyLegacyHBMCorrection_ && isRealData && runnum_ > 0)
    if (cell.subdet() == HcalBarrel) {
      const int ieta = cell.ieta();
      const int iphi = cell.iphi();
      corr = hbminus_special_ecorr(ieta, iphi, energy, runnum_);
    }
  return corr;
}

float SimpleHBHEPhase1Algo::m0Energy(const HBHEChannelInfo& info,
                                     const double fc_ampl,
                                     const bool applyContainmentCorrection,
                                     const double phaseNs,
                                     const int nSamplesToAdd) {
  int ibeg = static_cast<int>(info.soi()) + firstSampleShift_;
  if (ibeg < 0)
    ibeg = 0;
  double e = info.energyInWindow(ibeg, ibeg + nSamplesToAdd);

  // Pulse containment correction
  {
    double corrFactor = 1.0;
    if (applyContainmentCorrection)
      corrFactor = pulseCorr_.get(info.id(), nSamplesToAdd, phaseNs)->getCorrection(fc_ampl);
    e *= corrFactor;
  }

  return e;
}

float SimpleHBHEPhase1Algo::m0Time(const HBHEChannelInfo& info,
                                   const double fc_ampl,
                                   const int nSamplesToExamine) const {
  float time = -9999.f;  // historic value

  const unsigned nSamples = info.nSamples();
  if (nSamples > 2U) {
    const int soi = info.soi();
    int ibeg = soi + firstSampleShift_;
    if (ibeg < 0)
      ibeg = 0;
    const int iend = std::min(ibeg + nSamplesToExamine, (int)nSamples - 1);  // actual array

    unsigned maxI = info.peakEnergyTS((unsigned)ibeg, (unsigned)iend);  // requires unsigned params
    if (maxI < HBHEChannelInfo::MAXSAMPLES) {
      if (maxI >= nSamples)
        maxI = nSamples - 1U;  // just in case

      // Simplified evaluation for Phase1
      float emax0 = info.tsEnergy(maxI);
      float emax1 = 0.f;
      if (maxI < (nSamples - 1U))
        emax1 = info.tsEnergy(maxI + 1U);

      // consider soi reference for collisions
      int position = (int)maxI;
      if (nSamplesToExamine < (int)nSamples)
        position -= soi;

      time = 25.f * (float)position;
      if (emax0 > 0.f && emax1 > 0.f)
        time += 25.f * emax1 / (emax0 + emax1);  // 1st order corr.

      // TimeSlew correction
      time -= hcalTimeSlew_delay_->delay(std::max(1.0, fc_ampl), HcalTimeSlew::Medium);
    }
  }
  return time;
}
