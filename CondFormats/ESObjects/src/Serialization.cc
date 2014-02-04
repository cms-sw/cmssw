
#include "CondFormats/ESObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void ESADCToGeVConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ESvaluelow_);
    ar & BOOST_SERIALIZATION_NVP(ESvaluehigh_);
}
COND_SERIALIZATION_INSTANTIATE(ESADCToGeVConstant);

template <class Archive>
void ESChannelStatusCode::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(status_);
}
COND_SERIALIZATION_INSTANTIATE(ESChannelStatusCode);

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
COND_SERIALIZATION_INSTANTIATE(ESEEIntercalibConstants);

template <class Archive>
void ESGain::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(gain_);
}
COND_SERIALIZATION_INSTANTIATE(ESGain);

template <class Archive>
void ESMIPToGeVConstant::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ESvaluelow_);
    ar & BOOST_SERIALIZATION_NVP(ESvaluehigh_);
}
COND_SERIALIZATION_INSTANTIATE(ESMIPToGeVConstant);

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
COND_SERIALIZATION_INSTANTIATE(ESMissingEnergyCalibration);

template <class Archive>
void ESPedestal::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mean);
    ar & BOOST_SERIALIZATION_NVP(rms);
}
COND_SERIALIZATION_INSTANTIATE(ESPedestal);

template <class Archive>
void ESRecHitRatioCuts::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(r12Low_);
    ar & BOOST_SERIALIZATION_NVP(r23Low_);
    ar & BOOST_SERIALIZATION_NVP(r12High_);
    ar & BOOST_SERIALIZATION_NVP(r23High_);
}
COND_SERIALIZATION_INSTANTIATE(ESRecHitRatioCuts);

template <class Archive>
void ESStripGroupId::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(id_);
}
COND_SERIALIZATION_INSTANTIATE(ESStripGroupId);

template <class Archive>
void ESTBWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(map_);
}
COND_SERIALIZATION_INSTANTIATE(ESTBWeights);

template <class Archive>
void ESThresholds::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ts2_);
    ar & BOOST_SERIALIZATION_NVP(zs_);
}
COND_SERIALIZATION_INSTANTIATE(ESThresholds);

template <class Archive>
void ESTimeSampleWeights::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(w0_);
    ar & BOOST_SERIALIZATION_NVP(w1_);
    ar & BOOST_SERIALIZATION_NVP(w2_);
}
COND_SERIALIZATION_INSTANTIATE(ESTimeSampleWeights);

template <class Archive>
void ESWeightSet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(wgtBeforeSwitch_);
}
COND_SERIALIZATION_INSTANTIATE(ESWeightSet);

#include "CondFormats/ESObjects/src/SerializationManual.h"
