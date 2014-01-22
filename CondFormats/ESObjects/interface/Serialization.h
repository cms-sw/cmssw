#ifndef CondFormats_ESObjects_Serialization_H
#define CondFormats_ESObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void ESADCToGeVConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ESvaluelow_);
    ar & BOOST_SERIALIZATION_NVP(ESvaluehigh_);
}

template <class Archive>
void ESChannelStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}

template <typename T>
template <class Archive>
void ESCondObjectContainer<T>::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(es_);
}

template <class Archive>
void ESEEIntercalibConstants::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gammaLow0_);
    ar & BOOST_SERIALIZATION_NVP(alphaLow0_);
    ar & BOOST_SERIALIZATION_NVP(gammaHigh0_);
    ar & BOOST_SERIALIZATION_NVP(alphaHigh0_);
    ar & BOOST_SERIALIZATION_NVP(gammaLow1_);
    ar & BOOST_SERIALIZATION_NVP(alphaLow1_);
    ar & BOOST_SERIALIZATION_NVP(gammaHigh1_);
    ar & BOOST_SERIALIZATION_NVP(alphaHigh1_);
    ar & BOOST_SERIALIZATION_NVP(gammaLow2_);
    ar & BOOST_SERIALIZATION_NVP(alphaLow2_);
    ar & BOOST_SERIALIZATION_NVP(gammaHigh2_);
    ar & BOOST_SERIALIZATION_NVP(alphaHigh2_);
    ar & BOOST_SERIALIZATION_NVP(gammaLow3_);
    ar & BOOST_SERIALIZATION_NVP(alphaLow3_);
    ar & BOOST_SERIALIZATION_NVP(gammaHigh3_);
    ar & BOOST_SERIALIZATION_NVP(alphaHigh3_);
}

template <class Archive>
void ESGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain_);
}

template <class Archive>
void ESMIPToGeVConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ESvaluelow_);
    ar & BOOST_SERIALIZATION_NVP(ESvaluehigh_);
}

template <class Archive>
void ESMissingEnergyCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ConstAEta0_);
    ar & BOOST_SERIALIZATION_NVP(ConstBEta0_);
    ar & BOOST_SERIALIZATION_NVP(ConstAEta1_);
    ar & BOOST_SERIALIZATION_NVP(ConstBEta1_);
    ar & BOOST_SERIALIZATION_NVP(ConstAEta2_);
    ar & BOOST_SERIALIZATION_NVP(ConstBEta2_);
    ar & BOOST_SERIALIZATION_NVP(ConstAEta3_);
    ar & BOOST_SERIALIZATION_NVP(ConstBEta3_);
}

template <class Archive>
void ESPedestal::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(rms);
}

template <class Archive>
void ESRecHitRatioCuts::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(r12Low_);
    ar & BOOST_SERIALIZATION_NVP(r23Low_);
    ar & BOOST_SERIALIZATION_NVP(r12High_);
    ar & BOOST_SERIALIZATION_NVP(r23High_);
}

template <class Archive>
void ESStripGroupId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(id_);
}

template <class Archive>
void ESTBWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}

template <class Archive>
void ESThresholds::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ts2_);
    ar & BOOST_SERIALIZATION_NVP(zs_);
}

template <class Archive>
void ESTimeSampleWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(w0_);
    ar & BOOST_SERIALIZATION_NVP(w1_);
    ar & BOOST_SERIALIZATION_NVP(w2_);
}

template <class Archive>
void ESWeightSet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wgtBeforeSwitch_);
}

namespace cond {
namespace serialization {

template <>
struct access<ESADCToGeVConstant>
{
    static bool equal_(const ESADCToGeVConstant & first, const ESADCToGeVConstant & second)
    {
        return true
            and (equal(first.ESvaluelow_, second.ESvaluelow_))
            and (equal(first.ESvaluehigh_, second.ESvaluehigh_))
        ;
    }
};

template <>
struct access<ESChannelStatusCode>
{
    static bool equal_(const ESChannelStatusCode & first, const ESChannelStatusCode & second)
    {
        return true
            and (equal(first.status_, second.status_))
        ;
    }
};

template <typename T>
struct access<ESCondObjectContainer<T>>
{
    static bool equal_(const ESCondObjectContainer<T> & first, const ESCondObjectContainer<T> & second)
    {
        return true
            and (equal(first.es_, second.es_))
        ;
    }
};

template <>
struct access<ESEEIntercalibConstants>
{
    static bool equal_(const ESEEIntercalibConstants & first, const ESEEIntercalibConstants & second)
    {
        return true
            and (equal(first.gammaLow0_, second.gammaLow0_))
            and (equal(first.alphaLow0_, second.alphaLow0_))
            and (equal(first.gammaHigh0_, second.gammaHigh0_))
            and (equal(first.alphaHigh0_, second.alphaHigh0_))
            and (equal(first.gammaLow1_, second.gammaLow1_))
            and (equal(first.alphaLow1_, second.alphaLow1_))
            and (equal(first.gammaHigh1_, second.gammaHigh1_))
            and (equal(first.alphaHigh1_, second.alphaHigh1_))
            and (equal(first.gammaLow2_, second.gammaLow2_))
            and (equal(first.alphaLow2_, second.alphaLow2_))
            and (equal(first.gammaHigh2_, second.gammaHigh2_))
            and (equal(first.alphaHigh2_, second.alphaHigh2_))
            and (equal(first.gammaLow3_, second.gammaLow3_))
            and (equal(first.alphaLow3_, second.alphaLow3_))
            and (equal(first.gammaHigh3_, second.gammaHigh3_))
            and (equal(first.alphaHigh3_, second.alphaHigh3_))
        ;
    }
};

template <>
struct access<ESGain>
{
    static bool equal_(const ESGain & first, const ESGain & second)
    {
        return true
            and (equal(first.gain_, second.gain_))
        ;
    }
};

template <>
struct access<ESMIPToGeVConstant>
{
    static bool equal_(const ESMIPToGeVConstant & first, const ESMIPToGeVConstant & second)
    {
        return true
            and (equal(first.ESvaluelow_, second.ESvaluelow_))
            and (equal(first.ESvaluehigh_, second.ESvaluehigh_))
        ;
    }
};

template <>
struct access<ESMissingEnergyCalibration>
{
    static bool equal_(const ESMissingEnergyCalibration & first, const ESMissingEnergyCalibration & second)
    {
        return true
            and (equal(first.ConstAEta0_, second.ConstAEta0_))
            and (equal(first.ConstBEta0_, second.ConstBEta0_))
            and (equal(first.ConstAEta1_, second.ConstAEta1_))
            and (equal(first.ConstBEta1_, second.ConstBEta1_))
            and (equal(first.ConstAEta2_, second.ConstAEta2_))
            and (equal(first.ConstBEta2_, second.ConstBEta2_))
            and (equal(first.ConstAEta3_, second.ConstAEta3_))
            and (equal(first.ConstBEta3_, second.ConstBEta3_))
        ;
    }
};

template <>
struct access<ESPedestal>
{
    static bool equal_(const ESPedestal & first, const ESPedestal & second)
    {
        return true
            and (equal(first.mean, second.mean))
            and (equal(first.rms, second.rms))
        ;
    }
};

template <>
struct access<ESRecHitRatioCuts>
{
    static bool equal_(const ESRecHitRatioCuts & first, const ESRecHitRatioCuts & second)
    {
        return true
            and (equal(first.r12Low_, second.r12Low_))
            and (equal(first.r23Low_, second.r23Low_))
            and (equal(first.r12High_, second.r12High_))
            and (equal(first.r23High_, second.r23High_))
        ;
    }
};

template <>
struct access<ESStripGroupId>
{
    static bool equal_(const ESStripGroupId & first, const ESStripGroupId & second)
    {
        return true
            and (equal(first.id_, second.id_))
        ;
    }
};

template <>
struct access<ESTBWeights>
{
    static bool equal_(const ESTBWeights & first, const ESTBWeights & second)
    {
        return true
            and (equal(first.map_, second.map_))
        ;
    }
};

template <>
struct access<ESThresholds>
{
    static bool equal_(const ESThresholds & first, const ESThresholds & second)
    {
        return true
            and (equal(first.ts2_, second.ts2_))
            and (equal(first.zs_, second.zs_))
        ;
    }
};

template <>
struct access<ESTimeSampleWeights>
{
    static bool equal_(const ESTimeSampleWeights & first, const ESTimeSampleWeights & second)
    {
        return true
            and (equal(first.w0_, second.w0_))
            and (equal(first.w1_, second.w1_))
            and (equal(first.w2_, second.w2_))
        ;
    }
};

template <>
struct access<ESWeightSet>
{
    static bool equal_(const ESWeightSet & first, const ESWeightSet & second)
    {
        return true
            and (equal(first.wgtBeforeSwitch_, second.wgtBeforeSwitch_))
        ;
    }
};

}
}

#endif
