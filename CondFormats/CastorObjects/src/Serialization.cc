
#include "CondFormats/CastorObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void CastorCalibrationQIECoder::serialize(Archive & ar, const unsigned int)
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
COND_SERIALIZATION_INSTANTIATE(CastorCalibrationQIECoder);

template <class Archive>
void CastorCalibrationQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorCalibrationQIECoder>", boost::serialization::base_object<CastorCondObjectContainer<class CastorCalibrationQIECoder>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(CastorCalibrationQIEData);

template <class Archive>
void CastorChannelQuality::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorChannelStatus>", boost::serialization::base_object<CastorCondObjectContainer<class CastorChannelStatus>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(CastorChannelQuality);

template <class Archive>
void CastorChannelStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mStatus);
}
COND_SERIALIZATION_INSTANTIATE(CastorChannelStatus);

template <typename Item>
template <class Archive>
void CastorCondObjectContainer<Item>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(CASTORcontainer);
}

template <class Archive>
void CastorElectronicsMap::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mPItems);
    ar & BOOST_SERIALIZATION_NVP(mTItems);
}
COND_SERIALIZATION_INSTANTIATE(CastorElectronicsMap);

template <class Archive>
void CastorElectronicsMap::PrecisionItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}
COND_SERIALIZATION_INSTANTIATE(CastorElectronicsMap::PrecisionItem);

template <class Archive>
void CastorElectronicsMap::TriggerItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mTrigId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}
COND_SERIALIZATION_INSTANTIATE(CastorElectronicsMap::TriggerItem);

template <class Archive>
void CastorGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}
COND_SERIALIZATION_INSTANTIATE(CastorGain);

template <class Archive>
void CastorGainWidth::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}
COND_SERIALIZATION_INSTANTIATE(CastorGainWidth);

template <class Archive>
void CastorGainWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorGainWidth>", boost::serialization::base_object<CastorCondObjectContainer<class CastorGainWidth>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(CastorGainWidths);

template <class Archive>
void CastorGains::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorGain>", boost::serialization::base_object<CastorCondObjectContainer<class CastorGain>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(CastorGains);

template <class Archive>
void CastorPedestal::serialize(Archive & ar, const unsigned int)
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
COND_SERIALIZATION_INSTANTIATE(CastorPedestal);

template <class Archive>
void CastorPedestalWidth::serialize(Archive & ar, const unsigned int)
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
COND_SERIALIZATION_INSTANTIATE(CastorPedestalWidth);

template <class Archive>
void CastorPedestalWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorPedestalWidth>", boost::serialization::base_object<CastorCondObjectContainer<class CastorPedestalWidth>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}
COND_SERIALIZATION_INSTANTIATE(CastorPedestalWidths);

template <class Archive>
void CastorPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorPedestal>", boost::serialization::base_object<CastorCondObjectContainer<class CastorPedestal>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}
COND_SERIALIZATION_INSTANTIATE(CastorPedestals);

template <class Archive>
void CastorQIECoder::serialize(Archive & ar, const unsigned int)
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
COND_SERIALIZATION_INSTANTIATE(CastorQIECoder);

template <class Archive>
void CastorQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorQIECoder>", boost::serialization::base_object<CastorCondObjectContainer<class CastorQIECoder>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(CastorQIEData);

template <class Archive>
void CastorRecoParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mFirstSample);
    ar & BOOST_SERIALIZATION_NVP(mSamplesToAdd);
}
COND_SERIALIZATION_INSTANTIATE(CastorRecoParam);

template <class Archive>
void CastorRecoParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorRecoParam>", boost::serialization::base_object<CastorCondObjectContainer<class CastorRecoParam>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(CastorRecoParams);

template <class Archive>
void CastorSaturationCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mSatCorr);
}
COND_SERIALIZATION_INSTANTIATE(CastorSaturationCorr);

template <class Archive>
void CastorSaturationCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorSaturationCorr>", boost::serialization::base_object<CastorCondObjectContainer<class CastorSaturationCorr>>(*this));
}
COND_SERIALIZATION_INSTANTIATE(CastorSaturationCorrs);

#include "CondFormats/CastorObjects/src/SerializationManual.h"
