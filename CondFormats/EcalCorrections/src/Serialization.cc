
#include "CondFormats/EcalCorrections/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void EcalGlobalShowerContainmentCorrectionsVsEta::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(coefficients_);
}
COND_SERIALIZATION_INSTANTIATE(EcalGlobalShowerContainmentCorrectionsVsEta);

template <class Archive>
void EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}
COND_SERIALIZATION_INSTANTIATE(EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients);

template <class Archive>
void EcalShowerContainmentCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(groupmap_);
    ar & BOOST_SERIALIZATION_NVP(coefficients_);
}
COND_SERIALIZATION_INSTANTIATE(EcalShowerContainmentCorrections);

template <class Archive>
void EcalShowerContainmentCorrections::Coefficients::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}
COND_SERIALIZATION_INSTANTIATE(EcalShowerContainmentCorrections::Coefficients);

#include "CondFormats/EcalCorrections/src/SerializationManual.h"
