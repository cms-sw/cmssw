#ifndef CondFormats_HcalObjects_Serialization_H
#define CondFormats_HcalObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void HcalCalibrationQIECoder::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(bin0);
    ar & BOOST_SERIALIZATION_NVP(bin1);
    ar & BOOST_SERIALIZATION_NVP(bin2);
    ar & BOOST_SERIALIZATION_NVP(bin3);
    ar & BOOST_SERIALIZATION_NVP(bin4);
    ar & BOOST_SERIALIZATION_NVP(bin5);
    ar & BOOST_SERIALIZATION_NVP(bin6);
    ar & BOOST_SERIALIZATION_NVP(bin7);
    ar & BOOST_SERIALIZATION_NVP(bin8);
    ar & BOOST_SERIALIZATION_NVP(bin9);
    ar & BOOST_SERIALIZATION_NVP(bin10);
    ar & BOOST_SERIALIZATION_NVP(bin11);
    ar & BOOST_SERIALIZATION_NVP(bin12);
    ar & BOOST_SERIALIZATION_NVP(bin13);
    ar & BOOST_SERIALIZATION_NVP(bin14);
    ar & BOOST_SERIALIZATION_NVP(bin15);
    ar & BOOST_SERIALIZATION_NVP(bin16);
    ar & BOOST_SERIALIZATION_NVP(bin17);
    ar & BOOST_SERIALIZATION_NVP(bin18);
    ar & BOOST_SERIALIZATION_NVP(bin19);
    ar & BOOST_SERIALIZATION_NVP(bin20);
    ar & BOOST_SERIALIZATION_NVP(bin21);
    ar & BOOST_SERIALIZATION_NVP(bin22);
    ar & BOOST_SERIALIZATION_NVP(bin23);
    ar & BOOST_SERIALIZATION_NVP(bin24);
    ar & BOOST_SERIALIZATION_NVP(bin25);
    ar & BOOST_SERIALIZATION_NVP(bin26);
    ar & BOOST_SERIALIZATION_NVP(bin27);
    ar & BOOST_SERIALIZATION_NVP(bin28);
    ar & BOOST_SERIALIZATION_NVP(bin29);
    ar & BOOST_SERIALIZATION_NVP(bin30);
    ar & BOOST_SERIALIZATION_NVP(bin31);
}

template <class Archive>
void HcalCalibrationQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalCalibrationQIECoder>", boost::serialization::base_object<HcalCondObjectContainer<class HcalCalibrationQIECoder>>(*this));
}

template <class Archive>
void HcalChannelQuality::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalChannelStatus>", boost::serialization::base_object<HcalCondObjectContainer<class HcalChannelStatus>>(*this));
}

template <class Archive>
void HcalChannelStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mStatus);
}

template <class Archive>
void HcalCholeskyMatrices::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainerBase", boost::serialization::base_object<HcalCondObjectContainerBase>(*this));
    ar & BOOST_SERIALIZATION_NVP(HBcontainer);
    ar & BOOST_SERIALIZATION_NVP(HEcontainer);
    ar & BOOST_SERIALIZATION_NVP(HOcontainer);
    ar & BOOST_SERIALIZATION_NVP(HFcontainer);
}

template <class Archive>
void HcalCholeskyMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cmatrix);
    ar & BOOST_SERIALIZATION_NVP(mId);
}

template <typename Item>
template <class Archive>
void HcalCondObjectContainer<Item>::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainerBase", boost::serialization::base_object<HcalCondObjectContainerBase>(*this));
    ar & BOOST_SERIALIZATION_NVP(HBcontainer);
    ar & BOOST_SERIALIZATION_NVP(HEcontainer);
    ar & BOOST_SERIALIZATION_NVP(HOcontainer);
    ar & BOOST_SERIALIZATION_NVP(HFcontainer);
    ar & BOOST_SERIALIZATION_NVP(HTcontainer);
    ar & BOOST_SERIALIZATION_NVP(ZDCcontainer);
    ar & BOOST_SERIALIZATION_NVP(CALIBcontainer);
    ar & BOOST_SERIALIZATION_NVP(CASTORcontainer);
}

template <class Archive>
void HcalCondObjectContainerBase::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void HcalCovarianceMatrices::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainerBase", boost::serialization::base_object<HcalCondObjectContainerBase>(*this));
    ar & BOOST_SERIALIZATION_NVP(HBcontainer);
    ar & BOOST_SERIALIZATION_NVP(HEcontainer);
    ar & BOOST_SERIALIZATION_NVP(HOcontainer);
    ar & BOOST_SERIALIZATION_NVP(HFcontainer);
}

template <class Archive>
void HcalCovarianceMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(covariancematrix);
}

template <class Archive>
void HcalDcsMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mItems);
}

template <class Archive>
void HcalDcsMap::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mDcsId);
}

template <class Archive>
void HcalDcsValue::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mLS);
    ar & BOOST_SERIALIZATION_NVP(mValue);
    ar & BOOST_SERIALIZATION_NVP(mUpperLimit);
    ar & BOOST_SERIALIZATION_NVP(mLowerLimit);
}

template <class Archive>
void HcalDcsValues::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mHBValues);
    ar & BOOST_SERIALIZATION_NVP(mHBsorted);
    ar & BOOST_SERIALIZATION_NVP(mHEValues);
    ar & BOOST_SERIALIZATION_NVP(mHEsorted);
    ar & BOOST_SERIALIZATION_NVP(mHO0Values);
    ar & BOOST_SERIALIZATION_NVP(mHO0sorted);
    ar & BOOST_SERIALIZATION_NVP(mHO12Values);
    ar & BOOST_SERIALIZATION_NVP(mHO12sorted);
    ar & BOOST_SERIALIZATION_NVP(mHFValues);
    ar & BOOST_SERIALIZATION_NVP(mHFsorted);
}

template <class Archive>
void HcalElectronicsMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mPItems);
    ar & BOOST_SERIALIZATION_NVP(mTItems);
}

template <class Archive>
void HcalElectronicsMap::PrecisionItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}

template <class Archive>
void HcalElectronicsMap::TriggerItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mTrigId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}

template <class Archive>
void HcalFlagHFDigiTimeParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mHFdigiflagFirstSample);
    ar & BOOST_SERIALIZATION_NVP(mHFdigiflagSamplesToAdd);
    ar & BOOST_SERIALIZATION_NVP(mHFdigiflagExpectedPeak);
    ar & BOOST_SERIALIZATION_NVP(mHFdigiflagMinEthreshold);
    ar & BOOST_SERIALIZATION_NVP(mHFdigiflagCoefficients);
}

template <class Archive>
void HcalFlagHFDigiTimeParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalFlagHFDigiTimeParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalFlagHFDigiTimeParam>>(*this));
}

template <class Archive>
void HcalGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}

template <class Archive>
void HcalGainWidth::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}

template <class Archive>
void HcalGainWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalGainWidth>", boost::serialization::base_object<HcalCondObjectContainer<class HcalGainWidth>>(*this));
}

template <class Archive>
void HcalGains::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalGain>", boost::serialization::base_object<HcalCondObjectContainer<class HcalGain>>(*this));
}

template <class Archive>
void HcalL1TriggerObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mAvrgPed);
    ar & BOOST_SERIALIZATION_NVP(mRespCorrGain);
    ar & BOOST_SERIALIZATION_NVP(mFlag);
}

template <class Archive>
void HcalL1TriggerObjects::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalL1TriggerObject>", boost::serialization::base_object<HcalCondObjectContainer<class HcalL1TriggerObject>>(*this));
    ar & BOOST_SERIALIZATION_NVP(mTag);
    ar & BOOST_SERIALIZATION_NVP(mAlgo);
}

template <class Archive>
void HcalLUTCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}

template <class Archive>
void HcalLUTCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalLUTCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalLUTCorr>>(*this));
}

template <class Archive>
void HcalLongRecoParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mSignalTS);
    ar & BOOST_SERIALIZATION_NVP(mNoiseTS);
}

template <class Archive>
void HcalLongRecoParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalLongRecoParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalLongRecoParam>>(*this));
}

template <class Archive>
void HcalLutMetadata::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalLutMetadatum>", boost::serialization::base_object<HcalCondObjectContainer<class HcalLutMetadatum>>(*this));
    ar & BOOST_SERIALIZATION_NVP(mNonChannelData);
}

template <class Archive>
void HcalLutMetadata::NonChannelData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mRctLsb);
    ar & BOOST_SERIALIZATION_NVP(mNominalGain);
}

template <class Archive>
void HcalLutMetadatum::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mRCalib);
    ar & BOOST_SERIALIZATION_NVP(mLutGranularity);
    ar & BOOST_SERIALIZATION_NVP(mOutputLutThreshold);
}

template <class Archive>
void HcalMCParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mParam1);
}

template <class Archive>
void HcalMCParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalMCParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalMCParam>>(*this));
}

template <class Archive>
void HcalPFCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}

template <class Archive>
void HcalPFCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalPFCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalPFCorr>>(*this));
}

template <class Archive>
void HcalPedestal::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
    ar & BOOST_SERIALIZATION_NVP(mWidth0);
    ar & BOOST_SERIALIZATION_NVP(mWidth1);
    ar & BOOST_SERIALIZATION_NVP(mWidth2);
    ar & BOOST_SERIALIZATION_NVP(mWidth3);
}

template <class Archive>
void HcalPedestalWidth::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mSigma00);
    ar & BOOST_SERIALIZATION_NVP(mSigma01);
    ar & BOOST_SERIALIZATION_NVP(mSigma02);
    ar & BOOST_SERIALIZATION_NVP(mSigma03);
    ar & BOOST_SERIALIZATION_NVP(mSigma10);
    ar & BOOST_SERIALIZATION_NVP(mSigma11);
    ar & BOOST_SERIALIZATION_NVP(mSigma12);
    ar & BOOST_SERIALIZATION_NVP(mSigma13);
    ar & BOOST_SERIALIZATION_NVP(mSigma20);
    ar & BOOST_SERIALIZATION_NVP(mSigma21);
    ar & BOOST_SERIALIZATION_NVP(mSigma22);
    ar & BOOST_SERIALIZATION_NVP(mSigma23);
    ar & BOOST_SERIALIZATION_NVP(mSigma30);
    ar & BOOST_SERIALIZATION_NVP(mSigma31);
    ar & BOOST_SERIALIZATION_NVP(mSigma32);
    ar & BOOST_SERIALIZATION_NVP(mSigma33);
}

template <class Archive>
void HcalPedestalWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalPedestalWidth>", boost::serialization::base_object<HcalCondObjectContainer<class HcalPedestalWidth>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}

template <class Archive>
void HcalPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalPedestal>", boost::serialization::base_object<HcalCondObjectContainer<class HcalPedestal>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}

template <class Archive>
void HcalQIECoder::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mOffset00);
    ar & BOOST_SERIALIZATION_NVP(mOffset01);
    ar & BOOST_SERIALIZATION_NVP(mOffset02);
    ar & BOOST_SERIALIZATION_NVP(mOffset03);
    ar & BOOST_SERIALIZATION_NVP(mOffset10);
    ar & BOOST_SERIALIZATION_NVP(mOffset11);
    ar & BOOST_SERIALIZATION_NVP(mOffset12);
    ar & BOOST_SERIALIZATION_NVP(mOffset13);
    ar & BOOST_SERIALIZATION_NVP(mOffset20);
    ar & BOOST_SERIALIZATION_NVP(mOffset21);
    ar & BOOST_SERIALIZATION_NVP(mOffset22);
    ar & BOOST_SERIALIZATION_NVP(mOffset23);
    ar & BOOST_SERIALIZATION_NVP(mOffset30);
    ar & BOOST_SERIALIZATION_NVP(mOffset31);
    ar & BOOST_SERIALIZATION_NVP(mOffset32);
    ar & BOOST_SERIALIZATION_NVP(mOffset33);
    ar & BOOST_SERIALIZATION_NVP(mSlope00);
    ar & BOOST_SERIALIZATION_NVP(mSlope01);
    ar & BOOST_SERIALIZATION_NVP(mSlope02);
    ar & BOOST_SERIALIZATION_NVP(mSlope03);
    ar & BOOST_SERIALIZATION_NVP(mSlope10);
    ar & BOOST_SERIALIZATION_NVP(mSlope11);
    ar & BOOST_SERIALIZATION_NVP(mSlope12);
    ar & BOOST_SERIALIZATION_NVP(mSlope13);
    ar & BOOST_SERIALIZATION_NVP(mSlope20);
    ar & BOOST_SERIALIZATION_NVP(mSlope21);
    ar & BOOST_SERIALIZATION_NVP(mSlope22);
    ar & BOOST_SERIALIZATION_NVP(mSlope23);
    ar & BOOST_SERIALIZATION_NVP(mSlope30);
    ar & BOOST_SERIALIZATION_NVP(mSlope31);
    ar & BOOST_SERIALIZATION_NVP(mSlope32);
    ar & BOOST_SERIALIZATION_NVP(mSlope33);
}

template <class Archive>
void HcalQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalQIECoder>", boost::serialization::base_object<HcalCondObjectContainer<class HcalQIECoder>>(*this));
}

template <class Archive>
void HcalRecoParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mParam1);
    ar & BOOST_SERIALIZATION_NVP(mParam2);
}

template <class Archive>
void HcalRecoParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalRecoParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalRecoParam>>(*this));
}

template <class Archive>
void HcalRespCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}

template <class Archive>
void HcalRespCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalRespCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalRespCorr>>(*this));
}

template <class Archive>
void HcalTimeCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}

template <class Archive>
void HcalTimeCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalTimeCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalTimeCorr>>(*this));
}

template <class Archive>
void HcalTimingParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(m_nhits);
    ar & BOOST_SERIALIZATION_NVP(m_phase);
    ar & BOOST_SERIALIZATION_NVP(m_rms);
}

template <class Archive>
void HcalTimingParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalTimingParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalTimingParam>>(*this));
}

template <class Archive>
void HcalValidationCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}

template <class Archive>
void HcalValidationCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalValidationCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalValidationCorr>>(*this));
}

template <class Archive>
void HcalZSThreshold::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mLevel);
}

template <class Archive>
void HcalZSThresholds::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalZSThreshold>", boost::serialization::base_object<HcalCondObjectContainer<class HcalZSThreshold>>(*this));
}

namespace cond {
namespace serialization {

template <>
struct access<HcalCalibrationQIECoder>
{
    static bool equal_(const HcalCalibrationQIECoder & first, const HcalCalibrationQIECoder & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.bin0, second.bin0))
            and (equal(first.bin1, second.bin1))
            and (equal(first.bin2, second.bin2))
            and (equal(first.bin3, second.bin3))
            and (equal(first.bin4, second.bin4))
            and (equal(first.bin5, second.bin5))
            and (equal(first.bin6, second.bin6))
            and (equal(first.bin7, second.bin7))
            and (equal(first.bin8, second.bin8))
            and (equal(first.bin9, second.bin9))
            and (equal(first.bin10, second.bin10))
            and (equal(first.bin11, second.bin11))
            and (equal(first.bin12, second.bin12))
            and (equal(first.bin13, second.bin13))
            and (equal(first.bin14, second.bin14))
            and (equal(first.bin15, second.bin15))
            and (equal(first.bin16, second.bin16))
            and (equal(first.bin17, second.bin17))
            and (equal(first.bin18, second.bin18))
            and (equal(first.bin19, second.bin19))
            and (equal(first.bin20, second.bin20))
            and (equal(first.bin21, second.bin21))
            and (equal(first.bin22, second.bin22))
            and (equal(first.bin23, second.bin23))
            and (equal(first.bin24, second.bin24))
            and (equal(first.bin25, second.bin25))
            and (equal(first.bin26, second.bin26))
            and (equal(first.bin27, second.bin27))
            and (equal(first.bin28, second.bin28))
            and (equal(first.bin29, second.bin29))
            and (equal(first.bin30, second.bin30))
            and (equal(first.bin31, second.bin31))
        ;
    }
};

template <>
struct access<HcalCalibrationQIEData>
{
    static bool equal_(const HcalCalibrationQIEData & first, const HcalCalibrationQIEData & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalCalibrationQIECoder> &>(first), static_cast<const HcalCondObjectContainer<class HcalCalibrationQIECoder> &>(second)))
        ;
    }
};

template <>
struct access<HcalChannelQuality>
{
    static bool equal_(const HcalChannelQuality & first, const HcalChannelQuality & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalChannelStatus> &>(first), static_cast<const HcalCondObjectContainer<class HcalChannelStatus> &>(second)))
        ;
    }
};

template <>
struct access<HcalChannelStatus>
{
    static bool equal_(const HcalChannelStatus & first, const HcalChannelStatus & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mStatus, second.mStatus))
        ;
    }
};

template <>
struct access<HcalCholeskyMatrices>
{
    static bool equal_(const HcalCholeskyMatrices & first, const HcalCholeskyMatrices & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainerBase &>(first), static_cast<const HcalCondObjectContainerBase &>(second)))
            and (equal(first.HBcontainer, second.HBcontainer))
            and (equal(first.HEcontainer, second.HEcontainer))
            and (equal(first.HOcontainer, second.HOcontainer))
            and (equal(first.HFcontainer, second.HFcontainer))
        ;
    }
};

template <>
struct access<HcalCholeskyMatrix>
{
    static bool equal_(const HcalCholeskyMatrix & first, const HcalCholeskyMatrix & second)
    {
        return true
            and (equal(first.cmatrix, second.cmatrix))
            and (equal(first.mId, second.mId))
        ;
    }
};

template <typename Item>
struct access<HcalCondObjectContainer<Item>>
{
    static bool equal_(const HcalCondObjectContainer<Item> & first, const HcalCondObjectContainer<Item> & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainerBase &>(first), static_cast<const HcalCondObjectContainerBase &>(second)))
            and (equal(first.HBcontainer, second.HBcontainer))
            and (equal(first.HEcontainer, second.HEcontainer))
            and (equal(first.HOcontainer, second.HOcontainer))
            and (equal(first.HFcontainer, second.HFcontainer))
            and (equal(first.HTcontainer, second.HTcontainer))
            and (equal(first.ZDCcontainer, second.ZDCcontainer))
            and (equal(first.CALIBcontainer, second.CALIBcontainer))
            and (equal(first.CASTORcontainer, second.CASTORcontainer))
        ;
    }
};

template <>
struct access<HcalCondObjectContainerBase>
{
    static bool equal_(const HcalCondObjectContainerBase & first, const HcalCondObjectContainerBase & second)
    {
        return true
        ;
    }
};

template <>
struct access<HcalCovarianceMatrices>
{
    static bool equal_(const HcalCovarianceMatrices & first, const HcalCovarianceMatrices & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainerBase &>(first), static_cast<const HcalCondObjectContainerBase &>(second)))
            and (equal(first.HBcontainer, second.HBcontainer))
            and (equal(first.HEcontainer, second.HEcontainer))
            and (equal(first.HOcontainer, second.HOcontainer))
            and (equal(first.HFcontainer, second.HFcontainer))
        ;
    }
};

template <>
struct access<HcalCovarianceMatrix>
{
    static bool equal_(const HcalCovarianceMatrix & first, const HcalCovarianceMatrix & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.covariancematrix, second.covariancematrix))
        ;
    }
};

template <>
struct access<HcalDcsMap>
{
    static bool equal_(const HcalDcsMap & first, const HcalDcsMap & second)
    {
        return true
            and (equal(first.mItems, second.mItems))
        ;
    }
};

template <>
struct access<HcalDcsMap::Item>
{
    static bool equal_(const HcalDcsMap::Item & first, const HcalDcsMap::Item & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mDcsId, second.mDcsId))
        ;
    }
};

template <>
struct access<HcalDcsValue>
{
    static bool equal_(const HcalDcsValue & first, const HcalDcsValue & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mLS, second.mLS))
            and (equal(first.mValue, second.mValue))
            and (equal(first.mUpperLimit, second.mUpperLimit))
            and (equal(first.mLowerLimit, second.mLowerLimit))
        ;
    }
};

template <>
struct access<HcalDcsValues>
{
    static bool equal_(const HcalDcsValues & first, const HcalDcsValues & second)
    {
        return true
            and (equal(first.mHBValues, second.mHBValues))
            and (equal(first.mHBsorted, second.mHBsorted))
            and (equal(first.mHEValues, second.mHEValues))
            and (equal(first.mHEsorted, second.mHEsorted))
            and (equal(first.mHO0Values, second.mHO0Values))
            and (equal(first.mHO0sorted, second.mHO0sorted))
            and (equal(first.mHO12Values, second.mHO12Values))
            and (equal(first.mHO12sorted, second.mHO12sorted))
            and (equal(first.mHFValues, second.mHFValues))
            and (equal(first.mHFsorted, second.mHFsorted))
        ;
    }
};

template <>
struct access<HcalElectronicsMap>
{
    static bool equal_(const HcalElectronicsMap & first, const HcalElectronicsMap & second)
    {
        return true
            and (equal(first.mPItems, second.mPItems))
            and (equal(first.mTItems, second.mTItems))
        ;
    }
};

template <>
struct access<HcalElectronicsMap::PrecisionItem>
{
    static bool equal_(const HcalElectronicsMap::PrecisionItem & first, const HcalElectronicsMap::PrecisionItem & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mElId, second.mElId))
        ;
    }
};

template <>
struct access<HcalElectronicsMap::TriggerItem>
{
    static bool equal_(const HcalElectronicsMap::TriggerItem & first, const HcalElectronicsMap::TriggerItem & second)
    {
        return true
            and (equal(first.mTrigId, second.mTrigId))
            and (equal(first.mElId, second.mElId))
        ;
    }
};

template <>
struct access<HcalFlagHFDigiTimeParam>
{
    static bool equal_(const HcalFlagHFDigiTimeParam & first, const HcalFlagHFDigiTimeParam & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mHFdigiflagFirstSample, second.mHFdigiflagFirstSample))
            and (equal(first.mHFdigiflagSamplesToAdd, second.mHFdigiflagSamplesToAdd))
            and (equal(first.mHFdigiflagExpectedPeak, second.mHFdigiflagExpectedPeak))
            and (equal(first.mHFdigiflagMinEthreshold, second.mHFdigiflagMinEthreshold))
            and (equal(first.mHFdigiflagCoefficients, second.mHFdigiflagCoefficients))
        ;
    }
};

template <>
struct access<HcalFlagHFDigiTimeParams>
{
    static bool equal_(const HcalFlagHFDigiTimeParams & first, const HcalFlagHFDigiTimeParams & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalFlagHFDigiTimeParam> &>(first), static_cast<const HcalCondObjectContainer<class HcalFlagHFDigiTimeParam> &>(second)))
        ;
    }
};

template <>
struct access<HcalGain>
{
    static bool equal_(const HcalGain & first, const HcalGain & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue0, second.mValue0))
            and (equal(first.mValue1, second.mValue1))
            and (equal(first.mValue2, second.mValue2))
            and (equal(first.mValue3, second.mValue3))
        ;
    }
};

template <>
struct access<HcalGainWidth>
{
    static bool equal_(const HcalGainWidth & first, const HcalGainWidth & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue0, second.mValue0))
            and (equal(first.mValue1, second.mValue1))
            and (equal(first.mValue2, second.mValue2))
            and (equal(first.mValue3, second.mValue3))
        ;
    }
};

template <>
struct access<HcalGainWidths>
{
    static bool equal_(const HcalGainWidths & first, const HcalGainWidths & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalGainWidth> &>(first), static_cast<const HcalCondObjectContainer<class HcalGainWidth> &>(second)))
        ;
    }
};

template <>
struct access<HcalGains>
{
    static bool equal_(const HcalGains & first, const HcalGains & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalGain> &>(first), static_cast<const HcalCondObjectContainer<class HcalGain> &>(second)))
        ;
    }
};

template <>
struct access<HcalL1TriggerObject>
{
    static bool equal_(const HcalL1TriggerObject & first, const HcalL1TriggerObject & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mAvrgPed, second.mAvrgPed))
            and (equal(first.mRespCorrGain, second.mRespCorrGain))
            and (equal(first.mFlag, second.mFlag))
        ;
    }
};

template <>
struct access<HcalL1TriggerObjects>
{
    static bool equal_(const HcalL1TriggerObjects & first, const HcalL1TriggerObjects & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalL1TriggerObject> &>(first), static_cast<const HcalCondObjectContainer<class HcalL1TriggerObject> &>(second)))
            and (equal(first.mTag, second.mTag))
            and (equal(first.mAlgo, second.mAlgo))
        ;
    }
};

template <>
struct access<HcalLUTCorr>
{
    static bool equal_(const HcalLUTCorr & first, const HcalLUTCorr & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue, second.mValue))
        ;
    }
};

template <>
struct access<HcalLUTCorrs>
{
    static bool equal_(const HcalLUTCorrs & first, const HcalLUTCorrs & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalLUTCorr> &>(first), static_cast<const HcalCondObjectContainer<class HcalLUTCorr> &>(second)))
        ;
    }
};

template <>
struct access<HcalLongRecoParam>
{
    static bool equal_(const HcalLongRecoParam & first, const HcalLongRecoParam & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mSignalTS, second.mSignalTS))
            and (equal(first.mNoiseTS, second.mNoiseTS))
        ;
    }
};

template <>
struct access<HcalLongRecoParams>
{
    static bool equal_(const HcalLongRecoParams & first, const HcalLongRecoParams & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalLongRecoParam> &>(first), static_cast<const HcalCondObjectContainer<class HcalLongRecoParam> &>(second)))
        ;
    }
};

template <>
struct access<HcalLutMetadata>
{
    static bool equal_(const HcalLutMetadata & first, const HcalLutMetadata & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalLutMetadatum> &>(first), static_cast<const HcalCondObjectContainer<class HcalLutMetadatum> &>(second)))
            and (equal(first.mNonChannelData, second.mNonChannelData))
        ;
    }
};

template <>
struct access<HcalLutMetadata::NonChannelData>
{
    static bool equal_(const HcalLutMetadata::NonChannelData & first, const HcalLutMetadata::NonChannelData & second)
    {
        return true
            and (equal(first.mRctLsb, second.mRctLsb))
            and (equal(first.mNominalGain, second.mNominalGain))
        ;
    }
};

template <>
struct access<HcalLutMetadatum>
{
    static bool equal_(const HcalLutMetadatum & first, const HcalLutMetadatum & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mRCalib, second.mRCalib))
            and (equal(first.mLutGranularity, second.mLutGranularity))
            and (equal(first.mOutputLutThreshold, second.mOutputLutThreshold))
        ;
    }
};

template <>
struct access<HcalMCParam>
{
    static bool equal_(const HcalMCParam & first, const HcalMCParam & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mParam1, second.mParam1))
        ;
    }
};

template <>
struct access<HcalMCParams>
{
    static bool equal_(const HcalMCParams & first, const HcalMCParams & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalMCParam> &>(first), static_cast<const HcalCondObjectContainer<class HcalMCParam> &>(second)))
        ;
    }
};

template <>
struct access<HcalPFCorr>
{
    static bool equal_(const HcalPFCorr & first, const HcalPFCorr & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue, second.mValue))
        ;
    }
};

template <>
struct access<HcalPFCorrs>
{
    static bool equal_(const HcalPFCorrs & first, const HcalPFCorrs & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalPFCorr> &>(first), static_cast<const HcalCondObjectContainer<class HcalPFCorr> &>(second)))
        ;
    }
};

template <>
struct access<HcalPedestal>
{
    static bool equal_(const HcalPedestal & first, const HcalPedestal & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue0, second.mValue0))
            and (equal(first.mValue1, second.mValue1))
            and (equal(first.mValue2, second.mValue2))
            and (equal(first.mValue3, second.mValue3))
            and (equal(first.mWidth0, second.mWidth0))
            and (equal(first.mWidth1, second.mWidth1))
            and (equal(first.mWidth2, second.mWidth2))
            and (equal(first.mWidth3, second.mWidth3))
        ;
    }
};

template <>
struct access<HcalPedestalWidth>
{
    static bool equal_(const HcalPedestalWidth & first, const HcalPedestalWidth & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mSigma00, second.mSigma00))
            and (equal(first.mSigma01, second.mSigma01))
            and (equal(first.mSigma02, second.mSigma02))
            and (equal(first.mSigma03, second.mSigma03))
            and (equal(first.mSigma10, second.mSigma10))
            and (equal(first.mSigma11, second.mSigma11))
            and (equal(first.mSigma12, second.mSigma12))
            and (equal(first.mSigma13, second.mSigma13))
            and (equal(first.mSigma20, second.mSigma20))
            and (equal(first.mSigma21, second.mSigma21))
            and (equal(first.mSigma22, second.mSigma22))
            and (equal(first.mSigma23, second.mSigma23))
            and (equal(first.mSigma30, second.mSigma30))
            and (equal(first.mSigma31, second.mSigma31))
            and (equal(first.mSigma32, second.mSigma32))
            and (equal(first.mSigma33, second.mSigma33))
        ;
    }
};

template <>
struct access<HcalPedestalWidths>
{
    static bool equal_(const HcalPedestalWidths & first, const HcalPedestalWidths & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalPedestalWidth> &>(first), static_cast<const HcalCondObjectContainer<class HcalPedestalWidth> &>(second)))
            and (equal(first.unitIsADC, second.unitIsADC))
        ;
    }
};

template <>
struct access<HcalPedestals>
{
    static bool equal_(const HcalPedestals & first, const HcalPedestals & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalPedestal> &>(first), static_cast<const HcalCondObjectContainer<class HcalPedestal> &>(second)))
            and (equal(first.unitIsADC, second.unitIsADC))
        ;
    }
};

template <>
struct access<HcalQIECoder>
{
    static bool equal_(const HcalQIECoder & first, const HcalQIECoder & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mOffset00, second.mOffset00))
            and (equal(first.mOffset01, second.mOffset01))
            and (equal(first.mOffset02, second.mOffset02))
            and (equal(first.mOffset03, second.mOffset03))
            and (equal(first.mOffset10, second.mOffset10))
            and (equal(first.mOffset11, second.mOffset11))
            and (equal(first.mOffset12, second.mOffset12))
            and (equal(first.mOffset13, second.mOffset13))
            and (equal(first.mOffset20, second.mOffset20))
            and (equal(first.mOffset21, second.mOffset21))
            and (equal(first.mOffset22, second.mOffset22))
            and (equal(first.mOffset23, second.mOffset23))
            and (equal(first.mOffset30, second.mOffset30))
            and (equal(first.mOffset31, second.mOffset31))
            and (equal(first.mOffset32, second.mOffset32))
            and (equal(first.mOffset33, second.mOffset33))
            and (equal(first.mSlope00, second.mSlope00))
            and (equal(first.mSlope01, second.mSlope01))
            and (equal(first.mSlope02, second.mSlope02))
            and (equal(first.mSlope03, second.mSlope03))
            and (equal(first.mSlope10, second.mSlope10))
            and (equal(first.mSlope11, second.mSlope11))
            and (equal(first.mSlope12, second.mSlope12))
            and (equal(first.mSlope13, second.mSlope13))
            and (equal(first.mSlope20, second.mSlope20))
            and (equal(first.mSlope21, second.mSlope21))
            and (equal(first.mSlope22, second.mSlope22))
            and (equal(first.mSlope23, second.mSlope23))
            and (equal(first.mSlope30, second.mSlope30))
            and (equal(first.mSlope31, second.mSlope31))
            and (equal(first.mSlope32, second.mSlope32))
            and (equal(first.mSlope33, second.mSlope33))
        ;
    }
};

template <>
struct access<HcalQIEData>
{
    static bool equal_(const HcalQIEData & first, const HcalQIEData & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalQIECoder> &>(first), static_cast<const HcalCondObjectContainer<class HcalQIECoder> &>(second)))
        ;
    }
};

template <>
struct access<HcalRecoParam>
{
    static bool equal_(const HcalRecoParam & first, const HcalRecoParam & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mParam1, second.mParam1))
            and (equal(first.mParam2, second.mParam2))
        ;
    }
};

template <>
struct access<HcalRecoParams>
{
    static bool equal_(const HcalRecoParams & first, const HcalRecoParams & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalRecoParam> &>(first), static_cast<const HcalCondObjectContainer<class HcalRecoParam> &>(second)))
        ;
    }
};

template <>
struct access<HcalRespCorr>
{
    static bool equal_(const HcalRespCorr & first, const HcalRespCorr & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue, second.mValue))
        ;
    }
};

template <>
struct access<HcalRespCorrs>
{
    static bool equal_(const HcalRespCorrs & first, const HcalRespCorrs & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalRespCorr> &>(first), static_cast<const HcalCondObjectContainer<class HcalRespCorr> &>(second)))
        ;
    }
};

template <>
struct access<HcalTimeCorr>
{
    static bool equal_(const HcalTimeCorr & first, const HcalTimeCorr & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue, second.mValue))
        ;
    }
};

template <>
struct access<HcalTimeCorrs>
{
    static bool equal_(const HcalTimeCorrs & first, const HcalTimeCorrs & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalTimeCorr> &>(first), static_cast<const HcalCondObjectContainer<class HcalTimeCorr> &>(second)))
        ;
    }
};

template <>
struct access<HcalTimingParam>
{
    static bool equal_(const HcalTimingParam & first, const HcalTimingParam & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.m_nhits, second.m_nhits))
            and (equal(first.m_phase, second.m_phase))
            and (equal(first.m_rms, second.m_rms))
        ;
    }
};

template <>
struct access<HcalTimingParams>
{
    static bool equal_(const HcalTimingParams & first, const HcalTimingParams & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalTimingParam> &>(first), static_cast<const HcalCondObjectContainer<class HcalTimingParam> &>(second)))
        ;
    }
};

template <>
struct access<HcalValidationCorr>
{
    static bool equal_(const HcalValidationCorr & first, const HcalValidationCorr & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mValue, second.mValue))
        ;
    }
};

template <>
struct access<HcalValidationCorrs>
{
    static bool equal_(const HcalValidationCorrs & first, const HcalValidationCorrs & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalValidationCorr> &>(first), static_cast<const HcalCondObjectContainer<class HcalValidationCorr> &>(second)))
        ;
    }
};

template <>
struct access<HcalZSThreshold>
{
    static bool equal_(const HcalZSThreshold & first, const HcalZSThreshold & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mLevel, second.mLevel))
        ;
    }
};

template <>
struct access<HcalZSThresholds>
{
    static bool equal_(const HcalZSThresholds & first, const HcalZSThresholds & second)
    {
        return true
            and (equal(static_cast<const HcalCondObjectContainer<class HcalZSThreshold> &>(first), static_cast<const HcalCondObjectContainer<class HcalZSThreshold> &>(second)))
        ;
    }
};

}
}

#endif
