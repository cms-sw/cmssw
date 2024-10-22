#include <cfloat>

#include "CondTools/Hcal/interface/make_HFPhase1PMTParams.h"

#include "CondFormats/HcalObjects/interface/HcalDetIdTransform.h"
#include "CondFormats/HcalObjects/interface/HcalConstFunctor.h"

std::unique_ptr<HFPhase1PMTParams> make_HFPhase1PMTParams_dummy() {
  // Create "all pass" HFPhase1PMTData configuration
  HFPhase1PMTData::Cuts cuts;

  cuts[HFPhase1PMTData::T_0_MIN] = std::shared_ptr<AbsHcalFunctor>(new HcalConstFunctor(-FLT_MAX));
  cuts[HFPhase1PMTData::T_0_MAX] = std::shared_ptr<AbsHcalFunctor>(new HcalConstFunctor(FLT_MAX));
  cuts[HFPhase1PMTData::T_1_MIN] = cuts[HFPhase1PMTData::T_0_MIN];
  cuts[HFPhase1PMTData::T_1_MAX] = cuts[HFPhase1PMTData::T_0_MAX];
  cuts[HFPhase1PMTData::ASYMM_MIN] = cuts[HFPhase1PMTData::T_0_MIN];
  cuts[HFPhase1PMTData::ASYMM_MAX] = cuts[HFPhase1PMTData::T_0_MAX];

  std::unique_ptr<HFPhase1PMTData> defaultItem(new HFPhase1PMTData(cuts, -FLT_MAX, -FLT_MAX, FLT_MAX));

  // Other parts needed to create HFPhase1PMTParams
  const unsigned detIdTransformCode = HcalDetIdTransform::RAWID;
  HcalIndexLookup lookup;
  HcalItemColl<HFPhase1PMTData> coll;

  return std::unique_ptr<HFPhase1PMTParams>(
      new HFPhase1PMTParams(coll, lookup, detIdTransformCode, std::move(defaultItem)));
}
