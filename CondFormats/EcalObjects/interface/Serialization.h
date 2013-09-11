#ifndef CondFormats_EcalObjects_Serialization_H
#define CondFormats_EcalObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void EcalADCToGeVConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(EBvalue_);
    ar & BOOST_SERIALIZATION_NVP(EEvalue_);
}

template <class Archive>
void EcalChannelStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}

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

template <class Archive>
void EcalDCUTemperatures::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalDQMStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}

template <class Archive>
void EcalFunParams::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_params);
}

template <class Archive>
void EcalLaserAPDPNRatios::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(laser_map);
    ar & BOOST_SERIALIZATION_NVP(time_map);
}

template <class Archive>
void EcalLaserAPDPNRatios::EcalLaserAPDPNpair::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(p1);
    ar & BOOST_SERIALIZATION_NVP(p2);
    ar & BOOST_SERIALIZATION_NVP(p3);
}

template <class Archive>
void EcalLaserAPDPNRatios::EcalLaserTimeStamp::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(t1);
    ar & BOOST_SERIALIZATION_NVP(t2);
    ar & BOOST_SERIALIZATION_NVP(t3);
}

template <class Archive>
void EcalMGPAGainRatio::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain12Over6_);
    ar & BOOST_SERIALIZATION_NVP(gain6Over1_);
}

template <class Archive>
void EcalMappingElement::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(electronicsid);
    ar & BOOST_SERIALIZATION_NVP(triggerid);
}

template <class Archive>
void EcalPTMTemperatures::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

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

template <class Archive>
void EcalSampleMask::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(sampleMaskEB_);
    ar & BOOST_SERIALIZATION_NVP(sampleMaskEE_);
}

template <class Archive>
void EcalTBWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGCrystalStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}

template <class Archive>
void EcalTPGFineGrainConstEB::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ThresholdETLow_);
    ar & BOOST_SERIALIZATION_NVP(ThresholdETHigh_);
    ar & BOOST_SERIALIZATION_NVP(RatioLow_);
    ar & BOOST_SERIALIZATION_NVP(RatioHigh_);
    ar & BOOST_SERIALIZATION_NVP(LUT_);
}

template <class Archive>
void EcalTPGFineGrainEBGroup::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("EcalTPGGroups", boost::serialization::base_object<EcalTPGGroups>(*this));
}

template <class Archive>
void EcalTPGFineGrainEBIdMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGFineGrainStripEE::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGFineGrainStripEE::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(threshold);
    ar & BOOST_SERIALIZATION_NVP(lut);
}

template <class Archive>
void EcalTPGFineGrainTowerEE::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGGroups::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

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

template <class Archive>
void EcalTPGLut::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(lut_);
}

template <class Archive>
void EcalTPGLutGroup::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("EcalTPGGroups", boost::serialization::base_object<EcalTPGGroups>(*this));
}

template <class Archive>
void EcalTPGLutIdMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGPedestal::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mean_x12);
    ar & BOOST_SERIALIZATION_NVP(mean_x6);
    ar & BOOST_SERIALIZATION_NVP(mean_x1);
}

template <class Archive>
void EcalTPGPhysicsConst::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

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

template <class Archive>
void EcalTPGSlidingWindow::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGSpike::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGStripStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGTowerStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGWeightGroup::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("EcalTPGGroups", boost::serialization::base_object<EcalTPGGroups>(*this));
}

template <class Archive>
void EcalTPGWeightIdMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void EcalTPGWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(w0_);
    ar & BOOST_SERIALIZATION_NVP(w1_);
    ar & BOOST_SERIALIZATION_NVP(w2_);
    ar & BOOST_SERIALIZATION_NVP(w3_);
    ar & BOOST_SERIALIZATION_NVP(w4_);
}

template <class Archive>
void EcalTimeBiasCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(EBTimeCorrAmplitudeBins);
    ar & BOOST_SERIALIZATION_NVP(EBTimeCorrShiftBins);
    ar & BOOST_SERIALIZATION_NVP(EETimeCorrAmplitudeBins);
    ar & BOOST_SERIALIZATION_NVP(EETimeCorrShiftBins);
}

template <class Archive>
void EcalTimeDependentCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(value_map);
    ar & BOOST_SERIALIZATION_NVP(time_map);
}

template <class Archive>
void EcalTimeDependentCorrections::Times::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(t1);
    ar & BOOST_SERIALIZATION_NVP(t2);
    ar & BOOST_SERIALIZATION_NVP(t3);
}

template <class Archive>
void EcalTimeDependentCorrections::Values::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(p1);
    ar & BOOST_SERIALIZATION_NVP(p2);
    ar & BOOST_SERIALIZATION_NVP(p3);
}

template <class Archive>
void EcalTimeOffsetConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(EBvalue_);
    ar & BOOST_SERIALIZATION_NVP(EEvalue_);
}

template <class Archive>
void EcalWeightSet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wgtBeforeSwitch_);
    ar & BOOST_SERIALIZATION_NVP(wgtAfterSwitch_);
    ar & BOOST_SERIALIZATION_NVP(wgtChi2BeforeSwitch_);
    ar & BOOST_SERIALIZATION_NVP(wgtChi2AfterSwitch_);
}

template <class Archive>
void EcalXtalGroupId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(id_);
}

namespace cond {
namespace serialization {

template <>
struct access<EcalADCToGeVConstant>
{
    static bool equal_(const EcalADCToGeVConstant & first, const EcalADCToGeVConstant & second)
    {
        return true
            and (equal(first.EBvalue_, second.EBvalue_))
            and (equal(first.EEvalue_, second.EEvalue_))
        ;
    }
};

template <>
struct access<EcalChannelStatusCode>
{
    static bool equal_(const EcalChannelStatusCode & first, const EcalChannelStatusCode & second)
    {
        return true
            and (equal(first.status_, second.status_))
        ;
    }
};

template <typename T>
struct access<EcalCondObjectContainer<T>>
{
    static bool equal_(const EcalCondObjectContainer<T> & first, const EcalCondObjectContainer<T> & second)
    {
        return true
            and (equal(first.eb_, second.eb_))
            and (equal(first.ee_, second.ee_))
        ;
    }
};

template <typename T>
struct access<EcalCondTowerObjectContainer<T>>
{
    static bool equal_(const EcalCondTowerObjectContainer<T> & first, const EcalCondTowerObjectContainer<T> & second)
    {
        return true
            and (equal(first.eb_, second.eb_))
            and (equal(first.ee_, second.ee_))
        ;
    }
};

template <>
struct access<EcalDAQStatusCode>
{
    static bool equal_(const EcalDAQStatusCode & first, const EcalDAQStatusCode & second)
    {
        return true
            and (equal(first.status_, second.status_))
        ;
    }
};

template <>
struct access<EcalDCUTemperatures>
{
    static bool equal_(const EcalDCUTemperatures & first, const EcalDCUTemperatures & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalDQMStatusCode>
{
    static bool equal_(const EcalDQMStatusCode & first, const EcalDQMStatusCode & second)
    {
        return true
            and (equal(first.status_, second.status_))
        ;
    }
};

template <>
struct access<EcalFunParams>
{
    static bool equal_(const EcalFunParams & first, const EcalFunParams & second)
    {
        return true
            and (equal(first.m_params, second.m_params))
        ;
    }
};

template <>
struct access<EcalLaserAPDPNRatios>
{
    static bool equal_(const EcalLaserAPDPNRatios & first, const EcalLaserAPDPNRatios & second)
    {
        return true
            and (equal(first.laser_map, second.laser_map))
            and (equal(first.time_map, second.time_map))
        ;
    }
};

template <>
struct access<EcalLaserAPDPNRatios::EcalLaserAPDPNpair>
{
    static bool equal_(const EcalLaserAPDPNRatios::EcalLaserAPDPNpair & first, const EcalLaserAPDPNRatios::EcalLaserAPDPNpair & second)
    {
        return true
            and (equal(first.p1, second.p1))
            and (equal(first.p2, second.p2))
            and (equal(first.p3, second.p3))
        ;
    }
};

template <>
struct access<EcalLaserAPDPNRatios::EcalLaserTimeStamp>
{
    static bool equal_(const EcalLaserAPDPNRatios::EcalLaserTimeStamp & first, const EcalLaserAPDPNRatios::EcalLaserTimeStamp & second)
    {
        return true
            and (equal(first.t1, second.t1))
            and (equal(first.t2, second.t2))
            and (equal(first.t3, second.t3))
        ;
    }
};

template <>
struct access<EcalMGPAGainRatio>
{
    static bool equal_(const EcalMGPAGainRatio & first, const EcalMGPAGainRatio & second)
    {
        return true
            and (equal(first.gain12Over6_, second.gain12Over6_))
            and (equal(first.gain6Over1_, second.gain6Over1_))
        ;
    }
};

template <>
struct access<EcalMappingElement>
{
    static bool equal_(const EcalMappingElement & first, const EcalMappingElement & second)
    {
        return true
            and (equal(first.electronicsid, second.electronicsid))
            and (equal(first.triggerid, second.triggerid))
        ;
    }
};

template <>
struct access<EcalPTMTemperatures>
{
    static bool equal_(const EcalPTMTemperatures & first, const EcalPTMTemperatures & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalPedestal>
{
    static bool equal_(const EcalPedestal & first, const EcalPedestal & second)
    {
        return true
            and (equal(first.mean_x12, second.mean_x12))
            and (equal(first.rms_x12, second.rms_x12))
            and (equal(first.mean_x6, second.mean_x6))
            and (equal(first.rms_x6, second.rms_x6))
            and (equal(first.mean_x1, second.mean_x1))
            and (equal(first.rms_x1, second.rms_x1))
        ;
    }
};

template <>
struct access<EcalSRSettings>
{
    static bool equal_(const EcalSRSettings & first, const EcalSRSettings & second)
    {
        return true
            and (equal(first.deltaEta_, second.deltaEta_))
            and (equal(first.deltaPhi_, second.deltaPhi_))
            and (equal(first.ecalDccZs1stSample_, second.ecalDccZs1stSample_))
            and (equal(first.ebDccAdcToGeV_, second.ebDccAdcToGeV_))
            and (equal(first.eeDccAdcToGeV_, second.eeDccAdcToGeV_))
            and (equal(first.dccNormalizedWeights_, second.dccNormalizedWeights_))
            and (equal(first.symetricZS_, second.symetricZS_))
            and (equal(first.srpLowInterestChannelZS_, second.srpLowInterestChannelZS_))
            and (equal(first.srpHighInterestChannelZS_, second.srpHighInterestChannelZS_))
            and (equal(first.actions_, second.actions_))
            and (equal(first.tccMasksFromConfig_, second.tccMasksFromConfig_))
            and (equal(first.srpMasksFromConfig_, second.srpMasksFromConfig_))
            and (equal(first.dccMasks_, second.dccMasks_))
            and (equal(first.srfMasks_, second.srfMasks_))
            and (equal(first.substitutionSrfs_, second.substitutionSrfs_))
            and (equal(first.testerTccEmuSrpIds_, second.testerTccEmuSrpIds_))
            and (equal(first.testerSrpEmuSrpIds_, second.testerSrpEmuSrpIds_))
            and (equal(first.testerDccTestSrpIds_, second.testerDccTestSrpIds_))
            and (equal(first.testerSrpTestSrpIds_, second.testerSrpTestSrpIds_))
            and (equal(first.bxOffsets_, second.bxOffsets_))
            and (equal(first.bxGlobalOffset_, second.bxGlobalOffset_))
            and (equal(first.automaticMasks_, second.automaticMasks_))
            and (equal(first.automaticSrpSelect_, second.automaticSrpSelect_))
        ;
    }
};

template <>
struct access<EcalSampleMask>
{
    static bool equal_(const EcalSampleMask & first, const EcalSampleMask & second)
    {
        return true
            and (equal(first.sampleMaskEB_, second.sampleMaskEB_))
            and (equal(first.sampleMaskEE_, second.sampleMaskEE_))
        ;
    }
};

template <>
struct access<EcalTBWeights>
{
    static bool equal_(const EcalTBWeights & first, const EcalTBWeights & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGCrystalStatusCode>
{
    static bool equal_(const EcalTPGCrystalStatusCode & first, const EcalTPGCrystalStatusCode & second)
    {
        return true
            and (equal(first.status_, second.status_))
        ;
    }
};

template <>
struct access<EcalTPGFineGrainConstEB>
{
    static bool equal_(const EcalTPGFineGrainConstEB & first, const EcalTPGFineGrainConstEB & second)
    {
        return true
            and (equal(first.ThresholdETLow_, second.ThresholdETLow_))
            and (equal(first.ThresholdETHigh_, second.ThresholdETHigh_))
            and (equal(first.RatioLow_, second.RatioLow_))
            and (equal(first.RatioHigh_, second.RatioHigh_))
            and (equal(first.LUT_, second.LUT_))
        ;
    }
};

template <>
struct access<EcalTPGFineGrainEBGroup>
{
    static bool equal_(const EcalTPGFineGrainEBGroup & first, const EcalTPGFineGrainEBGroup & second)
    {
        return true
            and (equal(static_cast<const EcalTPGGroups &>(first), static_cast<const EcalTPGGroups &>(second)))
        ;
    }
};

template <>
struct access<EcalTPGFineGrainEBIdMap>
{
    static bool equal_(const EcalTPGFineGrainEBIdMap & first, const EcalTPGFineGrainEBIdMap & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGFineGrainStripEE>
{
    static bool equal_(const EcalTPGFineGrainStripEE & first, const EcalTPGFineGrainStripEE & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGFineGrainStripEE::Item>
{
    static bool equal_(const EcalTPGFineGrainStripEE::Item & first, const EcalTPGFineGrainStripEE::Item & second)
    {
        return true
            and (equal(first.threshold, second.threshold))
            and (equal(first.lut, second.lut))
        ;
    }
};

template <>
struct access<EcalTPGFineGrainTowerEE>
{
    static bool equal_(const EcalTPGFineGrainTowerEE & first, const EcalTPGFineGrainTowerEE & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGGroups>
{
    static bool equal_(const EcalTPGGroups & first, const EcalTPGGroups & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGLinearizationConstant>
{
    static bool equal_(const EcalTPGLinearizationConstant & first, const EcalTPGLinearizationConstant & second)
    {
        return true
            and (equal(first.mult_x12, second.mult_x12))
            and (equal(first.mult_x6, second.mult_x6))
            and (equal(first.mult_x1, second.mult_x1))
            and (equal(first.shift_x12, second.shift_x12))
            and (equal(first.shift_x6, second.shift_x6))
            and (equal(first.shift_x1, second.shift_x1))
        ;
    }
};

template <>
struct access<EcalTPGLut>
{
    static bool equal_(const EcalTPGLut & first, const EcalTPGLut & second)
    {
        return true
            and (equal(first.lut_, second.lut_))
        ;
    }
};

template <>
struct access<EcalTPGLutGroup>
{
    static bool equal_(const EcalTPGLutGroup & first, const EcalTPGLutGroup & second)
    {
        return true
            and (equal(static_cast<const EcalTPGGroups &>(first), static_cast<const EcalTPGGroups &>(second)))
        ;
    }
};

template <>
struct access<EcalTPGLutIdMap>
{
    static bool equal_(const EcalTPGLutIdMap & first, const EcalTPGLutIdMap & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGPedestal>
{
    static bool equal_(const EcalTPGPedestal & first, const EcalTPGPedestal & second)
    {
        return true
            and (equal(first.mean_x12, second.mean_x12))
            and (equal(first.mean_x6, second.mean_x6))
            and (equal(first.mean_x1, second.mean_x1))
        ;
    }
};

template <>
struct access<EcalTPGPhysicsConst>
{
    static bool equal_(const EcalTPGPhysicsConst & first, const EcalTPGPhysicsConst & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGPhysicsConst::Item>
{
    static bool equal_(const EcalTPGPhysicsConst::Item & first, const EcalTPGPhysicsConst::Item & second)
    {
        return true
            and (equal(first.EtSat, second.EtSat))
            and (equal(first.ttf_threshold_Low, second.ttf_threshold_Low))
            and (equal(first.ttf_threshold_High, second.ttf_threshold_High))
            and (equal(first.FG_lowThreshold, second.FG_lowThreshold))
            and (equal(first.FG_highThreshold, second.FG_highThreshold))
            and (equal(first.FG_lowRatio, second.FG_lowRatio))
            and (equal(first.FG_highRatio, second.FG_highRatio))
        ;
    }
};

template <>
struct access<EcalTPGSlidingWindow>
{
    static bool equal_(const EcalTPGSlidingWindow & first, const EcalTPGSlidingWindow & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGSpike>
{
    static bool equal_(const EcalTPGSpike & first, const EcalTPGSpike & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGStripStatus>
{
    static bool equal_(const EcalTPGStripStatus & first, const EcalTPGStripStatus & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGTowerStatus>
{
    static bool equal_(const EcalTPGTowerStatus & first, const EcalTPGTowerStatus & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGWeightGroup>
{
    static bool equal_(const EcalTPGWeightGroup & first, const EcalTPGWeightGroup & second)
    {
        return true
            and (equal(static_cast<const EcalTPGGroups &>(first), static_cast<const EcalTPGGroups &>(second)))
        ;
    }
};

template <>
struct access<EcalTPGWeightIdMap>
{
    static bool equal_(const EcalTPGWeightIdMap & first, const EcalTPGWeightIdMap & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<EcalTPGWeights>
{
    static bool equal_(const EcalTPGWeights & first, const EcalTPGWeights & second)
    {
        return true
            and (equal(first.w0_, second.w0_))
            and (equal(first.w1_, second.w1_))
            and (equal(first.w2_, second.w2_))
            and (equal(first.w3_, second.w3_))
            and (equal(first.w4_, second.w4_))
        ;
    }
};

template <>
struct access<EcalTimeBiasCorrections>
{
    static bool equal_(const EcalTimeBiasCorrections & first, const EcalTimeBiasCorrections & second)
    {
        return true
            and (equal(first.EBTimeCorrAmplitudeBins, second.EBTimeCorrAmplitudeBins))
            and (equal(first.EBTimeCorrShiftBins, second.EBTimeCorrShiftBins))
            and (equal(first.EETimeCorrAmplitudeBins, second.EETimeCorrAmplitudeBins))
            and (equal(first.EETimeCorrShiftBins, second.EETimeCorrShiftBins))
        ;
    }
};

template <>
struct access<EcalTimeDependentCorrections>
{
    static bool equal_(const EcalTimeDependentCorrections & first, const EcalTimeDependentCorrections & second)
    {
        return true
            and (equal(first.value_map, second.value_map))
            and (equal(first.time_map, second.time_map))
        ;
    }
};

template <>
struct access<EcalTimeDependentCorrections::Times>
{
    static bool equal_(const EcalTimeDependentCorrections::Times & first, const EcalTimeDependentCorrections::Times & second)
    {
        return true
            and (equal(first.t1, second.t1))
            and (equal(first.t2, second.t2))
            and (equal(first.t3, second.t3))
        ;
    }
};

template <>
struct access<EcalTimeDependentCorrections::Values>
{
    static bool equal_(const EcalTimeDependentCorrections::Values & first, const EcalTimeDependentCorrections::Values & second)
    {
        return true
            and (equal(first.p1, second.p1))
            and (equal(first.p2, second.p2))
            and (equal(first.p3, second.p3))
        ;
    }
};

template <>
struct access<EcalTimeOffsetConstant>
{
    static bool equal_(const EcalTimeOffsetConstant & first, const EcalTimeOffsetConstant & second)
    {
        return true
            and (equal(first.EBvalue_, second.EBvalue_))
            and (equal(first.EEvalue_, second.EEvalue_))
        ;
    }
};

template <>
struct access<EcalWeightSet>
{
    static bool equal_(const EcalWeightSet & first, const EcalWeightSet & second)
    {
        return true
            and (equal(first.wgtBeforeSwitch_, second.wgtBeforeSwitch_))
            and (equal(first.wgtAfterSwitch_, second.wgtAfterSwitch_))
            and (equal(first.wgtChi2BeforeSwitch_, second.wgtChi2BeforeSwitch_))
            and (equal(first.wgtChi2AfterSwitch_, second.wgtChi2AfterSwitch_))
        ;
    }
};

template <>
struct access<EcalXtalGroupId>
{
    static bool equal_(const EcalXtalGroupId & first, const EcalXtalGroupId & second)
    {
        return true
            and (equal(first.id_, second.id_))
        ;
    }
};

}
}

#endif
