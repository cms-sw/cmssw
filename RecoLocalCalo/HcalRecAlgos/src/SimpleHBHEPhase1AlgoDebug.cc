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

  const MahiFit* mahi = mahiOOTpuCorr_.get();
  if (mahi) {
    mahiOOTpuCorr_->setPulseShapeTemplate(theHcalPulseShapes_.getShape(info.recoShape()),hcalTimeSlewDelay);//,
    mahi->phase1Debug(info, mdi);
  }

  return mdi;
}
