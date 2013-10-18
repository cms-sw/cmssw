#ifndef CondFormats_CastorObjects_Serialization_H
#define CondFormats_CastorObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

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

template <class Archive>
void CastorCalibrationQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorCalibrationQIECoder>", boost::serialization::base_object<CastorCondObjectContainer<class CastorCalibrationQIECoder>>(*this));
}

template <class Archive>
void CastorChannelQuality::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorChannelStatus>", boost::serialization::base_object<CastorCondObjectContainer<class CastorChannelStatus>>(*this));
}

template <class Archive>
void CastorChannelStatus::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mStatus);
}

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

template <class Archive>
void CastorElectronicsMap::PrecisionItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}

template <class Archive>
void CastorElectronicsMap::TriggerItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mTrigId);
    ar & BOOST_SERIALIZATION_NVP(mElId);
}

template <class Archive>
void CastorGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}

template <class Archive>
void CastorGainWidth::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mValue0);
    ar & BOOST_SERIALIZATION_NVP(mValue1);
    ar & BOOST_SERIALIZATION_NVP(mValue2);
    ar & BOOST_SERIALIZATION_NVP(mValue3);
}

template <class Archive>
void CastorGainWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorGainWidth>", boost::serialization::base_object<CastorCondObjectContainer<class CastorGainWidth>>(*this));
}

template <class Archive>
void CastorGains::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorGain>", boost::serialization::base_object<CastorCondObjectContainer<class CastorGain>>(*this));
}

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

template <class Archive>
void CastorPedestalWidths::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorPedestalWidth>", boost::serialization::base_object<CastorCondObjectContainer<class CastorPedestalWidth>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}

template <class Archive>
void CastorPedestals::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorPedestal>", boost::serialization::base_object<CastorCondObjectContainer<class CastorPedestal>>(*this));
    ar & BOOST_SERIALIZATION_NVP(unitIsADC);
}

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

template <class Archive>
void CastorQIEData::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorQIECoder>", boost::serialization::base_object<CastorCondObjectContainer<class CastorQIECoder>>(*this));
}

template <class Archive>
void CastorRecoParam::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mFirstSample);
    ar & BOOST_SERIALIZATION_NVP(mSamplesToAdd);
}

template <class Archive>
void CastorRecoParams::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorRecoParam>", boost::serialization::base_object<CastorCondObjectContainer<class CastorRecoParam>>(*this));
}

template <class Archive>
void CastorSaturationCorr::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mId);
    ar & BOOST_SERIALIZATION_NVP(mSatCorr);
}

template <class Archive>
void CastorSaturationCorrs::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("CastorCondObjectContainer<class CastorSaturationCorr>", boost::serialization::base_object<CastorCondObjectContainer<class CastorSaturationCorr>>(*this));
}

namespace cond {
namespace serialization {

template <>
struct access<CastorCalibrationQIECoder>
{
    static bool equal_(const CastorCalibrationQIECoder & first, const CastorCalibrationQIECoder & second)
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
struct access<CastorCalibrationQIEData>
{
    static bool equal_(const CastorCalibrationQIEData & first, const CastorCalibrationQIEData & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorCalibrationQIECoder> &>(first), static_cast<const CastorCondObjectContainer<class CastorCalibrationQIECoder> &>(second)))
        ;
    }
};

template <>
struct access<CastorChannelQuality>
{
    static bool equal_(const CastorChannelQuality & first, const CastorChannelQuality & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorChannelStatus> &>(first), static_cast<const CastorCondObjectContainer<class CastorChannelStatus> &>(second)))
        ;
    }
};

template <>
struct access<CastorChannelStatus>
{
    static bool equal_(const CastorChannelStatus & first, const CastorChannelStatus & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mStatus, second.mStatus))
        ;
    }
};

template <typename Item>
struct access<CastorCondObjectContainer<Item>>
{
    static bool equal_(const CastorCondObjectContainer<Item> & first, const CastorCondObjectContainer<Item> & second)
    {
        return true
            and (equal(first.CASTORcontainer, second.CASTORcontainer))
        ;
    }
};

template <>
struct access<CastorElectronicsMap>
{
    static bool equal_(const CastorElectronicsMap & first, const CastorElectronicsMap & second)
    {
        return true
            and (equal(first.mPItems, second.mPItems))
            and (equal(first.mTItems, second.mTItems))
        ;
    }
};

template <>
struct access<CastorElectronicsMap::PrecisionItem>
{
    static bool equal_(const CastorElectronicsMap::PrecisionItem & first, const CastorElectronicsMap::PrecisionItem & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mElId, second.mElId))
        ;
    }
};

template <>
struct access<CastorElectronicsMap::TriggerItem>
{
    static bool equal_(const CastorElectronicsMap::TriggerItem & first, const CastorElectronicsMap::TriggerItem & second)
    {
        return true
            and (equal(first.mTrigId, second.mTrigId))
            and (equal(first.mElId, second.mElId))
        ;
    }
};

template <>
struct access<CastorGain>
{
    static bool equal_(const CastorGain & first, const CastorGain & second)
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
struct access<CastorGainWidth>
{
    static bool equal_(const CastorGainWidth & first, const CastorGainWidth & second)
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
struct access<CastorGainWidths>
{
    static bool equal_(const CastorGainWidths & first, const CastorGainWidths & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorGainWidth> &>(first), static_cast<const CastorCondObjectContainer<class CastorGainWidth> &>(second)))
        ;
    }
};

template <>
struct access<CastorGains>
{
    static bool equal_(const CastorGains & first, const CastorGains & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorGain> &>(first), static_cast<const CastorCondObjectContainer<class CastorGain> &>(second)))
        ;
    }
};

template <>
struct access<CastorPedestal>
{
    static bool equal_(const CastorPedestal & first, const CastorPedestal & second)
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
struct access<CastorPedestalWidth>
{
    static bool equal_(const CastorPedestalWidth & first, const CastorPedestalWidth & second)
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
struct access<CastorPedestalWidths>
{
    static bool equal_(const CastorPedestalWidths & first, const CastorPedestalWidths & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorPedestalWidth> &>(first), static_cast<const CastorCondObjectContainer<class CastorPedestalWidth> &>(second)))
            and (equal(first.unitIsADC, second.unitIsADC))
        ;
    }
};

template <>
struct access<CastorPedestals>
{
    static bool equal_(const CastorPedestals & first, const CastorPedestals & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorPedestal> &>(first), static_cast<const CastorCondObjectContainer<class CastorPedestal> &>(second)))
            and (equal(first.unitIsADC, second.unitIsADC))
        ;
    }
};

template <>
struct access<CastorQIECoder>
{
    static bool equal_(const CastorQIECoder & first, const CastorQIECoder & second)
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
struct access<CastorQIEData>
{
    static bool equal_(const CastorQIEData & first, const CastorQIEData & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorQIECoder> &>(first), static_cast<const CastorCondObjectContainer<class CastorQIECoder> &>(second)))
        ;
    }
};

template <>
struct access<CastorRecoParam>
{
    static bool equal_(const CastorRecoParam & first, const CastorRecoParam & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mFirstSample, second.mFirstSample))
            and (equal(first.mSamplesToAdd, second.mSamplesToAdd))
        ;
    }
};

template <>
struct access<CastorRecoParams>
{
    static bool equal_(const CastorRecoParams & first, const CastorRecoParams & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorRecoParam> &>(first), static_cast<const CastorCondObjectContainer<class CastorRecoParam> &>(second)))
        ;
    }
};

template <>
struct access<CastorSaturationCorr>
{
    static bool equal_(const CastorSaturationCorr & first, const CastorSaturationCorr & second)
    {
        return true
            and (equal(first.mId, second.mId))
            and (equal(first.mSatCorr, second.mSatCorr))
        ;
    }
};

template <>
struct access<CastorSaturationCorrs>
{
    static bool equal_(const CastorSaturationCorrs & first, const CastorSaturationCorrs & second)
    {
        return true
            and (equal(static_cast<const CastorCondObjectContainer<class CastorSaturationCorr> &>(first), static_cast<const CastorCondObjectContainer<class CastorSaturationCorr> &>(second)))
        ;
    }
};

}
}

#endif
