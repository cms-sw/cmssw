#ifndef RecoLocalCalo_HcalRecAlgos_SimpleHBHEPhase1Algo_h_
#define RecoLocalCalo_HcalRecAlgos_SimpleHBHEPhase1Algo_h_

#include <memory>
#include <vector>

// Base class header
#include "RecoLocalCalo/HcalRecAlgos/interface/AbsHBHEPhase1Algo.h"

// Other headers
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/PulseShapeFitOOTPileupCorrection.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalDeterministicFit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/MahiFit.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

class SimpleHBHEPhase1Algo : public AbsHBHEPhase1Algo {
public:
  // Constructor arguments:
  //
  //   firstSampleShift -- first TS w.r.t. SOI to use for "Method 0"
  //                       reconstruction.
  //
  //   samplesToAdd     -- default number of samples to add for "Method 0"
  //                       reconstruction. If, let say, SOI = 4,
  //                       firstSampleShift = -1, and samplesToAdd = 3
  //                       then the code will add time slices 3, 4, and 5.
  //
  //   phaseNS          -- default "phase" parameter for the pulse
  //                       containment correction
  //
  //   timeShift        -- time shift for QIE11 TDC times
  //
  //   correctForPhaseContainment -- default switch for applying pulse
  //                                 containment correction for "Method 0"
  //
  //   m2               -- "Method 2" object
  //
  //   detFit           -- "Method 3" (a.k.a. "deterministic fit") object
  //
  SimpleHBHEPhase1Algo(int firstSampleShift,
                       int samplesToAdd,
                       float phaseNS,
                       float timeShift,
                       bool correctForPhaseContainment,
                       bool applyLegacyHBMCorrection,
                       std::unique_ptr<PulseShapeFitOOTPileupCorrection> m2,
                       std::unique_ptr<HcalDeterministicFit> detFit,
                       std::unique_ptr<MahiFit> mahi,
                       edm::ConsumesCollector iC);

  inline ~SimpleHBHEPhase1Algo() override {}

  // Methods to override from the base class
  void beginRun(const edm::Run&, const edm::EventSetup&) override;
  void endRun() override;

  inline bool isConfigurable() const override { return false; }

  HBHERecHit reconstruct(const HBHEChannelInfo& info,
                         const HcalRecoParam* params,
                         const HcalCalibrations& calibs,
                         bool isRealData) override;
  // Basic accessors
  inline int getFirstSampleShift() const { return firstSampleShift_; }
  inline int getSamplesToAdd() const { return samplesToAdd_; }
  inline float getPhaseNS() const { return phaseNS_; }
  inline float getTimeShift() const { return timeShift_; }
  inline bool isCorrectingForPhaseContainment() const { return corrFPC_; }
  inline int getRunNumber() const { return runnum_; }

  edm::ESGetToken<HcalTimeSlew, HcalTimeSlewRecord> delayToken_;
  const HcalTimeSlew* hcalTimeSlew_delay_;

protected:
  // Special HB- correction
  float hbminusCorrectionFactor(const HcalDetId& cell, float energy, bool isRealData) const;

  // "Method 0" rechit energy. Calls a non-const member of
  // HcalPulseContainmentManager, so no const qualifier here.
  // HB- correction is not applied inside this function.
  float m0Energy(const HBHEChannelInfo& info,
                 double reconstructedCharge,
                 bool applyContainmentCorrection,
                 double phaseNS,
                 int nSamplesToAdd);

  // "Method 0" rechit timing (original low-pileup QIE8 algorithm)
  float m0Time(const HBHEChannelInfo& info, double reconstructedCharge, int nSamplesToExamine) const;

private:
  HcalPulseContainmentManager pulseCorr_;

  int firstSampleShift_;
  int samplesToAdd_;
  float phaseNS_;
  float timeShift_;
  int runnum_;
  bool corrFPC_;
  bool applyLegacyHBMCorrection_;

  // "Metod 2" algorithm
  std::unique_ptr<PulseShapeFitOOTPileupCorrection> psFitOOTpuCorr_;

  // "Metod 3" algorithm
  std::unique_ptr<HcalDeterministicFit> hltOOTpuCorr_;

  // Mahi algorithm
  std::unique_ptr<MahiFit> mahiOOTpuCorr_;

  HcalPulseShapes theHcalPulseShapes_;
};

#endif  // RecoLocalCalo_HcalRecAlgos_SimpleHBHEPhase1Algo_h_
