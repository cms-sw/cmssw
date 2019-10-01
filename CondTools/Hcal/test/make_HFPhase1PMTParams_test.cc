#include <cfloat>

#include "CondTools/Hcal/interface/make_HFPhase1PMTParams.h"

#include "CondFormats/HcalObjects/interface/HcalDetIdTransform.h"
#include "CondFormats/HcalObjects/interface/HcalConstFunctor.h"
#include "CondFormats/HcalObjects/interface/HcalPiecewiseLinearFunctor.h"
#include "CondFormats/HcalObjects/interface/HcalLinearCompositionFunctor.h"

#if BOOST_VERSION < 105600 || BOOST_VERSION > 105800
#ifdef BAD_BOOST_VERSION
#undef BAD_BOOST_VERSION
#endif
#else
#define BAD_BOOST_VERSION
#endif

std::unique_ptr<HFPhase1PMTParams> make_HFPhase1PMTParams_test() {
  typedef std::pair<double, double> P;

  // Create HFPhase1PMTData configuration for testing
  // cut visualization code. Note that the cut values
  // used are absolutely not realistic.
  HFPhase1PMTData::Cuts cuts;

  const double minCharge0 = 10.0;
  const double minCharge1 = 20.0;
  const double minChargeAsymm = 30.0;

  std::vector<P> pts;
  pts.push_back(P(minCharge0, 10.0));
  pts.push_back(P(10.0 * minCharge0, 12.0));
  pts.push_back(P(20.0 * minCharge0, 13.0));
  cuts[HFPhase1PMTData::T_0_MIN] = std::shared_ptr<AbsHcalFunctor>(new HcalPiecewiseLinearFunctor(pts, false, false));

  pts.clear();
  pts.push_back(P(minCharge0, 20.0));
  pts.push_back(P(10.0 * minCharge0, 18.0));
  pts.push_back(P(20.0 * minCharge0, 17.0));
  cuts[HFPhase1PMTData::T_0_MAX] = std::shared_ptr<AbsHcalFunctor>(new HcalPiecewiseLinearFunctor(pts, false, false));

#ifdef BAD_BOOST_VERSION
  pts.clear();
  pts.push_back(P(minCharge0, 10.0 - 2.0));
  pts.push_back(P(10.0 * minCharge0, 12.0 - 2.0));
  pts.push_back(P(20.0 * minCharge0, 13.0 - 2.0));
  cuts[HFPhase1PMTData::T_1_MIN] = std::shared_ptr<AbsHcalFunctor>(new HcalPiecewiseLinearFunctor(pts, false, false));
#else
  cuts[HFPhase1PMTData::T_1_MIN] =
      std::shared_ptr<AbsHcalFunctor>(new HcalLinearCompositionFunctor(cuts[HFPhase1PMTData::T_0_MIN], 1.0, -2.0));
#endif

#ifdef BAD_BOOST_VERSION
  pts.clear();
  pts.push_back(P(minCharge0, 20.0 + 2.0));
  pts.push_back(P(10.0 * minCharge0, 18.0 + 2.0));
  pts.push_back(P(20.0 * minCharge0, 17.0 + 2.0));
  cuts[HFPhase1PMTData::T_1_MAX] = std::shared_ptr<AbsHcalFunctor>(new HcalPiecewiseLinearFunctor(pts, false, false));
#else
  cuts[HFPhase1PMTData::T_1_MAX] =
      std::shared_ptr<AbsHcalFunctor>(new HcalLinearCompositionFunctor(cuts[HFPhase1PMTData::T_0_MAX], 1.0, 2.0));
#endif

  pts.clear();
  pts.push_back(P(5.0 * minChargeAsymm, 2.0));
  pts.push_back(P(10.0 * minChargeAsymm, 1.0));
  cuts[HFPhase1PMTData::ASYMM_MAX] = std::shared_ptr<AbsHcalFunctor>(new HcalPiecewiseLinearFunctor(pts, false, false));

#ifdef BAD_BOOST_VERSION
  pts.clear();
  pts.push_back(P(5.0 * minChargeAsymm, -2.0));
  pts.push_back(P(10.0 * minChargeAsymm, -1.0));
  cuts[HFPhase1PMTData::ASYMM_MIN] = std::shared_ptr<AbsHcalFunctor>(new HcalPiecewiseLinearFunctor(pts, false, false));
#else
  cuts[HFPhase1PMTData::ASYMM_MIN] =
      std::shared_ptr<AbsHcalFunctor>(new HcalLinearCompositionFunctor(cuts[HFPhase1PMTData::ASYMM_MAX], -1.0, 0.0));
#endif

  std::unique_ptr<HFPhase1PMTData> defaultItem(new HFPhase1PMTData(cuts, minCharge0, minCharge1, minChargeAsymm));

  // Other parts needed to create HFPhase1PMTParams
  const unsigned detIdTransformCode = HcalDetIdTransform::RAWID;
  HcalIndexLookup lookup;
  HcalItemColl<HFPhase1PMTData> coll;

  HcalDetId id(HcalForward, 29, 5, 1);
  lookup.add(HcalDetIdTransform::transform(id, detIdTransformCode), 0);

#ifdef BAD_BOOST_VERSION
  pts.clear();
  pts.push_back(P(5.0 * minChargeAsymm, -2.0 + 1.0));
  pts.push_back(P(10.0 * minChargeAsymm, -1.0 + 1.0));
  cuts[HFPhase1PMTData::ASYMM_MIN] = std::shared_ptr<AbsHcalFunctor>(new HcalPiecewiseLinearFunctor(pts, false, false));
#else
  cuts[HFPhase1PMTData::ASYMM_MIN] =
      std::shared_ptr<AbsHcalFunctor>(new HcalLinearCompositionFunctor(cuts[HFPhase1PMTData::ASYMM_MAX], -1.0, 1.0));
#endif

  std::unique_ptr<HFPhase1PMTData> firstItem(new HFPhase1PMTData(cuts, minCharge0, minCharge1, minChargeAsymm));
  coll.push_back(std::move(firstItem));

  return std::unique_ptr<HFPhase1PMTParams>(
      new HFPhase1PMTParams(coll, lookup, detIdTransformCode, std::move(defaultItem)));
}
