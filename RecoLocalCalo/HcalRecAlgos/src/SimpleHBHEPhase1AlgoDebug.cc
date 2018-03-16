#include <algorithm>
#include <iostream>

#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/SimpleHBHEPhase1AlgoDebug.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCorrectionFunctions.h"

#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHitAuxSetter.h"
#include "DataFormats/METReco/interface/HcalPhase1FlagLabels.h"
#include "CondFormats/DataRecord/interface/HcalTimeSlewRecord.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

MahiDebugInfo SimpleHBHEPhase1AlgoDebug::recoDebug(const HBHEChannelInfo& info,
						   const bool isData,
						   const HcalTimeSlew *hcalTimeSlewDelay) 
{
  MahiDebugInfo mdi;
  const HcalDetId channelId(info.id());
  
  // Calculate "Method 0" quantities
  //float m0t = 0.f, m0E = 0.f;
  //{
  //  int ibeg = static_cast<int>(info.soi()) + firstSampleShift_;
  //  if (ibeg < 0)
  //    ibeg = 0;
  //  const int nSamplesToAdd = params ? params->samplesToAdd() : samplesToAdd_;
  //  const double fc_ampl = info.chargeInWindow(ibeg, ibeg + nSamplesToAdd);
  //  const bool applyContainment = params ? params->correctForPhaseContainment() : corrFPC_;
  //  const float phasens = params ? params->correctionPhaseNS() : phaseNS_;
  //  m0E = m0Energy(info, fc_ampl, applyContainment, phasens, nSamplesToAdd);
  //  m0E *= hbminusCorrectionFactor(channelId, m0E, isData);
  //  m0t = m0Time(info, fc_ampl, calibs, nSamplesToAdd);
  //}
  //
  //// Run "Method 2"
  //float m2t = 0.f, m2E = 0.f, chi2 = -1.f;
  //bool useTriple = false;
  //const PulseShapeFitOOTPileupCorrection* method2 = psFitOOTpuCorr_.get();
  //if (method2)
  //  {
  //    psFitOOTpuCorr_->setPulseShapeTemplate(theHcalPulseShapes_.getShape(info.recoShape()),
  //					     !info.hasTimeInfo(),info.nSamples());
  //    // "phase1Apply" call below sets m2E, m2t, useTriple, and chi2.
  //    // These parameters are pased by non-const reference.
  //    method2->phase1Apply(info, m2E, m2t, useTriple, chi2, hcalTimeSlew_delay_);
  //    m2E *= hbminusCorrectionFactor(channelId, m2E, isData);
  //  }
  //
  //// Run "Method 3"
  //float m3t = 0.f, m3E = 0.f;
  //const HcalDeterministicFit* method3 = hltOOTpuCorr_.get();
  //if (method3)
  //  {
  //    // "phase1Apply" sets m3E and m3t (pased by non-const reference)
  //    method3->phase1Apply(info, m3E, m3t, hcalTimeSlew_delay_);
  //    m3E *= hbminusCorrectionFactor(channelId, m3E, isData);
  //  }
  
  // Run Mahi
  //float m4E = 0.f, m4chi2 = -1.f;
  //float m4T = 0.f;
  //bool m4UseTriple=false;

  const MahiFit* mahi = mahiOOTpuCorr_.get();
  if (mahi) {
    mahiOOTpuCorr_->setPulseShapeTemplate(theHcalPulseShapes_.getShape(info.recoShape()),hcalTimeSlewDelay);//,
    mahi->phase1Debug(info, mdi);
  }

  return mdi;
}
