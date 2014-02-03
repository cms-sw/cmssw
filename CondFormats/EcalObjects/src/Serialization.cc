
#include "CondFormats/EcalObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void EcalADCToGeVConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(EBvalue_);
    ar & BOOST_SERIALIZATION_NVP(EEvalue_);
}
COND_SERIALIZATION_INSTANTIATE(EcalADCToGeVConstant);

template <class Archive>
void EcalChannelStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}
COND_SERIALIZATION_INSTANTIATE(EcalChannelStatusCode);

template <typename T>
template <class Archive>
void EcalCondObjectContainer<T>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(eb_);
    ar & BOOST_SERIALIZATION_NVP(ee_);
}

template <typename T>
template <class Archive>
void EcalCondTowerObjectContainer<T>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(eb_);
    ar & BOOST_SERIALIZATION_NVP(ee_);
}

template <class Archive>
void EcalDAQStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}
COND_SERIALIZATION_INSTANTIATE(EcalDAQStatusCode);

template <class Archive>
void EcalDCUTemperatures::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalDCUTemperatures);

template <class Archive>
void EcalDQMStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}
COND_SERIALIZATION_INSTANTIATE(EcalDQMStatusCode);

template <class Archive>
void EcalFunParams::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_params);
}
COND_SERIALIZATION_INSTANTIATE(EcalFunParams);

template <class Archive>
void EcalLaserAPDPNRatios::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(laser_map);
    ar & BOOST_SERIALIZATION_NVP(time_map);
}
COND_SERIALIZATION_INSTANTIATE(EcalLaserAPDPNRatios);

template <class Archive>
void EcalLaserAPDPNRatios::EcalLaserAPDPNpair::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(p1);
    ar & BOOST_SERIALIZATION_NVP(p2);
    ar & BOOST_SERIALIZATION_NVP(p3);
}
COND_SERIALIZATION_INSTANTIATE(EcalLaserAPDPNRatios::EcalLaserAPDPNpair);

template <class Archive>
void EcalLaserAPDPNRatios::EcalLaserTimeStamp::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(t1);
    ar & BOOST_SERIALIZATION_NVP(t2);
    ar & BOOST_SERIALIZATION_NVP(t3);
}
COND_SERIALIZATION_INSTANTIATE(EcalLaserAPDPNRatios::EcalLaserTimeStamp);

template <class Archive>
void EcalMGPAGainRatio::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain12Over6_);
    ar & BOOST_SERIALIZATION_NVP(gain6Over1_);
}
COND_SERIALIZATION_INSTANTIATE(EcalMGPAGainRatio);

template <class Archive>
void EcalMappingElement::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(electronicsid);
    ar & BOOST_SERIALIZATION_NVP(triggerid);
}
COND_SERIALIZATION_INSTANTIATE(EcalMappingElement);

template <class Archive>
void EcalPTMTemperatures::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalPTMTemperatures);

template <class Archive>
void EcalPedestal::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mean_x12);
    ar & BOOST_SERIALIZATION_NVP(rms_x12);
    ar & BOOST_SERIALIZATION_NVP(mean_x6);
    ar & BOOST_SERIALIZATION_NVP(rms_x6);
    ar & BOOST_SERIALIZATION_NVP(mean_x1);
    ar & BOOST_SERIALIZATION_NVP(rms_x1);
}
COND_SERIALIZATION_INSTANTIATE(EcalPedestal);

template <class Archive>
void EcalSRSettings::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(deltaEta_);
    ar & BOOST_SERIALIZATION_NVP(deltaPhi_);
    ar & BOOST_SERIALIZATION_NVP(ecalDccZs1stSample_);
    ar & BOOST_SERIALIZATION_NVP(ebDccAdcToGeV_);
    ar & BOOST_SERIALIZATION_NVP(eeDccAdcToGeV_);
    ar & BOOST_SERIALIZATION_NVP(dccNormalizedWeights_);
    ar & BOOST_SERIALIZATION_NVP(symetricZS_);
    ar & BOOST_SERIALIZATION_NVP(srpLowInterestChannelZS_);
    ar & BOOST_SERIALIZATION_NVP(srpHighInterestChannelZS_);
    ar & BOOST_SERIALIZATION_NVP(actions_);
    ar & BOOST_SERIALIZATION_NVP(tccMasksFromConfig_);
    ar & BOOST_SERIALIZATION_NVP(srpMasksFromConfig_);
    ar & BOOST_SERIALIZATION_NVP(dccMasks_);
    ar & BOOST_SERIALIZATION_NVP(srfMasks_);
    ar & BOOST_SERIALIZATION_NVP(substitutionSrfs_);
    ar & BOOST_SERIALIZATION_NVP(testerTccEmuSrpIds_);
    ar & BOOST_SERIALIZATION_NVP(testerSrpEmuSrpIds_);
    ar & BOOST_SERIALIZATION_NVP(testerDccTestSrpIds_);
    ar & BOOST_SERIALIZATION_NVP(testerSrpTestSrpIds_);
    ar & BOOST_SERIALIZATION_NVP(bxOffsets_);
    ar & BOOST_SERIALIZATION_NVP(bxGlobalOffset_);
    ar & BOOST_SERIALIZATION_NVP(automaticMasks_);
    ar & BOOST_SERIALIZATION_NVP(automaticSrpSelect_);
}
COND_SERIALIZATION_INSTANTIATE(EcalSRSettings);

template <class Archive>
void EcalSampleMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(sampleMaskEB_);
    ar & BOOST_SERIALIZATION_NVP(sampleMaskEE_);
}
COND_SERIALIZATION_INSTANTIATE(EcalSampleMask);

template <class Archive>
void EcalTBWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTBWeights);

template <class Archive>
void EcalTPGCrystalStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGCrystalStatusCode);

template <class Archive>
void EcalTPGFineGrainConstEB::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ThresholdETLow_);
    ar & BOOST_SERIALIZATION_NVP(ThresholdETHigh_);
    ar & BOOST_SERIALIZATION_NVP(RatioLow_);
    ar & BOOST_SERIALIZATION_NVP(RatioHigh_);
    ar & BOOST_SERIALIZATION_NVP(LUT_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGFineGrainConstEB);

template <class Archive>
void EcalTPGFineGrainEBGroup::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("EcalTPGGroups", boost::serialization::base_object<EcalTPGGroups>(*this));
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGFineGrainEBGroup);

template <class Archive>
void EcalTPGFineGrainEBIdMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGFineGrainEBIdMap);

template <class Archive>
void EcalTPGFineGrainStripEE::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGFineGrainStripEE);

template <class Archive>
void EcalTPGFineGrainStripEE::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(threshold);
    ar & BOOST_SERIALIZATION_NVP(lut);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGFineGrainStripEE::Item);

template <class Archive>
void EcalTPGFineGrainTowerEE::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGFineGrainTowerEE);

template <class Archive>
void EcalTPGGroups::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGGroups);

template <class Archive>
void EcalTPGLinearizationConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mult_x12);
    ar & BOOST_SERIALIZATION_NVP(mult_x6);
    ar & BOOST_SERIALIZATION_NVP(mult_x1);
    ar & BOOST_SERIALIZATION_NVP(shift_x12);
    ar & BOOST_SERIALIZATION_NVP(shift_x6);
    ar & BOOST_SERIALIZATION_NVP(shift_x1);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGLinearizationConstant);

template <class Archive>
void EcalTPGLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(lut_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGLut);

template <class Archive>
void EcalTPGLutGroup::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("EcalTPGGroups", boost::serialization::base_object<EcalTPGGroups>(*this));
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGLutGroup);

template <class Archive>
void EcalTPGLutIdMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGLutIdMap);

template <class Archive>
void EcalTPGPedestal::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mean_x12);
    ar & BOOST_SERIALIZATION_NVP(mean_x6);
    ar & BOOST_SERIALIZATION_NVP(mean_x1);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGPedestal);

template <class Archive>
void EcalTPGPhysicsConst::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGPhysicsConst);

template <class Archive>
void EcalTPGPhysicsConst::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(EtSat);
    ar & BOOST_SERIALIZATION_NVP(ttf_threshold_Low);
    ar & BOOST_SERIALIZATION_NVP(ttf_threshold_High);
    ar & BOOST_SERIALIZATION_NVP(FG_lowThreshold);
    ar & BOOST_SERIALIZATION_NVP(FG_highThreshold);
    ar & BOOST_SERIALIZATION_NVP(FG_lowRatio);
    ar & BOOST_SERIALIZATION_NVP(FG_highRatio);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGPhysicsConst::Item);

template <class Archive>
void EcalTPGSlidingWindow::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGSlidingWindow);

template <class Archive>
void EcalTPGSpike::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGSpike);

template <class Archive>
void EcalTPGStripStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGStripStatus);

template <class Archive>
void EcalTPGTowerStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGTowerStatus);

template <class Archive>
void EcalTPGWeightGroup::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("EcalTPGGroups", boost::serialization::base_object<EcalTPGGroups>(*this));
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGWeightGroup);

template <class Archive>
void EcalTPGWeightIdMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGWeightIdMap);

template <class Archive>
void EcalTPGWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(w0_);
    ar & BOOST_SERIALIZATION_NVP(w1_);
    ar & BOOST_SERIALIZATION_NVP(w2_);
    ar & BOOST_SERIALIZATION_NVP(w3_);
    ar & BOOST_SERIALIZATION_NVP(w4_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTPGWeights);

template <class Archive>
void EcalTimeBiasCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(EBTimeCorrAmplitudeBins);
    ar & BOOST_SERIALIZATION_NVP(EBTimeCorrShiftBins);
    ar & BOOST_SERIALIZATION_NVP(EETimeCorrAmplitudeBins);
    ar & BOOST_SERIALIZATION_NVP(EETimeCorrShiftBins);
}
COND_SERIALIZATION_INSTANTIATE(EcalTimeBiasCorrections);

template <class Archive>
void EcalTimeDependentCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(value_map);
    ar & BOOST_SERIALIZATION_NVP(time_map);
}
COND_SERIALIZATION_INSTANTIATE(EcalTimeDependentCorrections);

template <class Archive>
void EcalTimeDependentCorrections::Times::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(t1);
    ar & BOOST_SERIALIZATION_NVP(t2);
    ar & BOOST_SERIALIZATION_NVP(t3);
}
COND_SERIALIZATION_INSTANTIATE(EcalTimeDependentCorrections::Times);

template <class Archive>
void EcalTimeDependentCorrections::Values::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(p1);
    ar & BOOST_SERIALIZATION_NVP(p2);
    ar & BOOST_SERIALIZATION_NVP(p3);
}
COND_SERIALIZATION_INSTANTIATE(EcalTimeDependentCorrections::Values);

template <class Archive>
void EcalTimeOffsetConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(EBvalue_);
    ar & BOOST_SERIALIZATION_NVP(EEvalue_);
}
COND_SERIALIZATION_INSTANTIATE(EcalTimeOffsetConstant);

template <class Archive>
void EcalWeightSet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wgtBeforeSwitch_);
    ar & BOOST_SERIALIZATION_NVP(wgtAfterSwitch_);
    ar & BOOST_SERIALIZATION_NVP(wgtChi2BeforeSwitch_);
    ar & BOOST_SERIALIZATION_NVP(wgtChi2AfterSwitch_);
}
COND_SERIALIZATION_INSTANTIATE(EcalWeightSet);

template <class Archive>
void EcalXtalGroupId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(id_);
}
COND_SERIALIZATION_INSTANTIATE(EcalXtalGroupId);

#include "CondFormats/EcalObjects/src/SerializationManual.h"
