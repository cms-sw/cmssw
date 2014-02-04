
#include "CondFormats/HcalObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

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
COND_SERIALIZATION_INSTANTIATE(HcalCalibrationQIECoder);

template <class Archive>
void HcalCalibrationQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalCalibrationQIECoder>", boost::serialization::base_object<HcalCondObjectContainer<class HcalCalibrationQIECoder>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalCalibrationQIEData);

template <class Archive>
void HcalChannelQuality::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalChannelStatus>", boost::serialization::base_object<HcalCondObjectContainer<class HcalChannelStatus>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalChannelQuality);

template <class Archive>
void HcalChannelStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mStatus);
}
COND_SERIALIZATION_INSTANTIATE(HcalChannelStatus);

template <class Archive>
void HcalCholeskyMatrices::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainerBase", boost::serialization::base_object<HcalCondObjectContainerBase>(*this));
    ar & BOOST_SERIALIZATION_NVP(HBcontainer);
    ar & BOOST_SERIALIZATION_NVP(HEcontainer);
    ar & BOOST_SERIALIZATION_NVP(HOcontainer);
    ar & BOOST_SERIALIZATION_NVP(HFcontainer);
}
COND_SERIALIZATION_INSTANTIATE(HcalCholeskyMatrices);

template <class Archive>
void HcalCholeskyMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(cmatrix);
    ar & BOOST_SERIALIZATION_NVP(mId);
}
COND_SERIALIZATION_INSTANTIATE(HcalCholeskyMatrix);

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
COND_SERIALIZATION_INSTANTIATE(HcalCondObjectContainerBase);

template <class Archive>
void HcalCovarianceMatrices::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainerBase", boost::serialization::base_object<HcalCondObjectContainerBase>(*this));
    ar & BOOST_SERIALIZATION_NVP(HBcontainer);
    ar & BOOST_SERIALIZATION_NVP(HEcontainer);
    ar & BOOST_SERIALIZATION_NVP(HOcontainer);
    ar & BOOST_SERIALIZATION_NVP(HFcontainer);
}
COND_SERIALIZATION_INSTANTIATE(HcalCovarianceMatrices);

template <class Archive>
void HcalCovarianceMatrix::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(covariancematrix);
}
COND_SERIALIZATION_INSTANTIATE(HcalCovarianceMatrix);

template <class Archive>
void HcalDcsMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mItems);
}
COND_SERIALIZATION_INSTANTIATE(HcalDcsMap);

template <class Archive>
void HcalDcsMap::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mDcsId);
}
COND_SERIALIZATION_INSTANTIATE(HcalDcsMap::Item);

template <class Archive>
void HcalDcsValue::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mLS);
    ar & BOOST_SERIALIZATION_NVP(mValue);
    ar & BOOST_SERIALIZATION_NVP(mUpperLimit);
    ar & BOOST_SERIALIZATION_NVP(mLowerLimit);
}
COND_SERIALIZATION_INSTANTIATE(HcalDcsValue);

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
COND_SERIALIZATION_INSTANTIATE(HcalDcsValues);

template <class Archive>
void HcalElectronicsMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mPItems);
    ar & BOOST_SERIALIZATION_NVP(mTItems);
}
COND_SERIALIZATION_INSTANTIATE(HcalElectronicsMap);

template <class Archive>
void HcalElectronicsMap::PrecisionItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}
COND_SERIALIZATION_INSTANTIATE(HcalElectronicsMap::PrecisionItem);

template <class Archive>
void HcalElectronicsMap::TriggerItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mTrigId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}
COND_SERIALIZATION_INSTANTIATE(HcalElectronicsMap::TriggerItem);

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
COND_SERIALIZATION_INSTANTIATE(HcalFlagHFDigiTimeParam);

template <class Archive>
void HcalFlagHFDigiTimeParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalFlagHFDigiTimeParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalFlagHFDigiTimeParam>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalFlagHFDigiTimeParams);

template <class Archive>
void HcalGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}
COND_SERIALIZATION_INSTANTIATE(HcalGain);

template <class Archive>
void HcalGainWidth::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}
COND_SERIALIZATION_INSTANTIATE(HcalGainWidth);

template <class Archive>
void HcalGainWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalGainWidth>", boost::serialization::base_object<HcalCondObjectContainer<class HcalGainWidth>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalGainWidths);

template <class Archive>
void HcalGains::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalGain>", boost::serialization::base_object<HcalCondObjectContainer<class HcalGain>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalGains);

template <class Archive>
void HcalL1TriggerObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mAvrgPed);
    ar & BOOST_SERIALIZATION_NVP(mRespCorrGain);
    ar & BOOST_SERIALIZATION_NVP(mFlag);
}
COND_SERIALIZATION_INSTANTIATE(HcalL1TriggerObject);

template <class Archive>
void HcalL1TriggerObjects::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalL1TriggerObject>", boost::serialization::base_object<HcalCondObjectContainer<class HcalL1TriggerObject>>(*this));
    ar & BOOST_SERIALIZATION_NVP(mTag);
    ar & BOOST_SERIALIZATION_NVP(mAlgo);
}
COND_SERIALIZATION_INSTANTIATE(HcalL1TriggerObjects);

template <class Archive>
void HcalLUTCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}
COND_SERIALIZATION_INSTANTIATE(HcalLUTCorr);

template <class Archive>
void HcalLUTCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalLUTCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalLUTCorr>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalLUTCorrs);

template <class Archive>
void HcalLongRecoParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mSignalTS);
    ar & BOOST_SERIALIZATION_NVP(mNoiseTS);
}
COND_SERIALIZATION_INSTANTIATE(HcalLongRecoParam);

template <class Archive>
void HcalLongRecoParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalLongRecoParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalLongRecoParam>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalLongRecoParams);

template <class Archive>
void HcalLutMetadata::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalLutMetadatum>", boost::serialization::base_object<HcalCondObjectContainer<class HcalLutMetadatum>>(*this));
    ar & BOOST_SERIALIZATION_NVP(mNonChannelData);
}
COND_SERIALIZATION_INSTANTIATE(HcalLutMetadata);

template <class Archive>
void HcalLutMetadata::NonChannelData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mRctLsb);
    ar & BOOST_SERIALIZATION_NVP(mNominalGain);
}
COND_SERIALIZATION_INSTANTIATE(HcalLutMetadata::NonChannelData);

template <class Archive>
void HcalLutMetadatum::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mRCalib);
    ar & BOOST_SERIALIZATION_NVP(mLutGranularity);
    ar & BOOST_SERIALIZATION_NVP(mOutputLutThreshold);
}
COND_SERIALIZATION_INSTANTIATE(HcalLutMetadatum);

template <class Archive>
void HcalMCParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mParam1);
}
COND_SERIALIZATION_INSTANTIATE(HcalMCParam);

template <class Archive>
void HcalMCParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalMCParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalMCParam>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalMCParams);

template <class Archive>
void HcalPFCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}
COND_SERIALIZATION_INSTANTIATE(HcalPFCorr);

template <class Archive>
void HcalPFCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalPFCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalPFCorr>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalPFCorrs);

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
COND_SERIALIZATION_INSTANTIATE(HcalPedestal);

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
COND_SERIALIZATION_INSTANTIATE(HcalPedestalWidth);

template <class Archive>
void HcalPedestalWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalPedestalWidth>", boost::serialization::base_object<HcalCondObjectContainer<class HcalPedestalWidth>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}
COND_SERIALIZATION_INSTANTIATE(HcalPedestalWidths);

template <class Archive>
void HcalPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalPedestal>", boost::serialization::base_object<HcalCondObjectContainer<class HcalPedestal>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}
COND_SERIALIZATION_INSTANTIATE(HcalPedestals);

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
COND_SERIALIZATION_INSTANTIATE(HcalQIECoder);

template <class Archive>
void HcalQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalQIECoder>", boost::serialization::base_object<HcalCondObjectContainer<class HcalQIECoder>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalQIEData);

template <class Archive>
void HcalRecoParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mParam1);
    ar & BOOST_SERIALIZATION_NVP(mParam2);
}
COND_SERIALIZATION_INSTANTIATE(HcalRecoParam);

template <class Archive>
void HcalRecoParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalRecoParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalRecoParam>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalRecoParams);

template <class Archive>
void HcalRespCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}
COND_SERIALIZATION_INSTANTIATE(HcalRespCorr);

template <class Archive>
void HcalRespCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalRespCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalRespCorr>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalRespCorrs);

template <class Archive>
void HcalTimeCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}
COND_SERIALIZATION_INSTANTIATE(HcalTimeCorr);

template <class Archive>
void HcalTimeCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalTimeCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalTimeCorr>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalTimeCorrs);

template <class Archive>
void HcalTimingParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(m_nhits);
    ar & BOOST_SERIALIZATION_NVP(m_phase);
    ar & BOOST_SERIALIZATION_NVP(m_rms);
}
COND_SERIALIZATION_INSTANTIATE(HcalTimingParam);

template <class Archive>
void HcalTimingParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalTimingParam>", boost::serialization::base_object<HcalCondObjectContainer<class HcalTimingParam>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalTimingParams);

template <class Archive>
void HcalValidationCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue);
}
COND_SERIALIZATION_INSTANTIATE(HcalValidationCorr);

template <class Archive>
void HcalValidationCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalValidationCorr>", boost::serialization::base_object<HcalCondObjectContainer<class HcalValidationCorr>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalValidationCorrs);

template <class Archive>
void HcalZSThreshold::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mLevel);
}
COND_SERIALIZATION_INSTANTIATE(HcalZSThreshold);

template <class Archive>
void HcalZSThresholds::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("HcalCondObjectContainer<class HcalZSThreshold>", boost::serialization::base_object<HcalCondObjectContainer<class HcalZSThreshold>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(HcalZSThresholds);

#include "CondFormats/HcalObjects/src/SerializationManual.h"
