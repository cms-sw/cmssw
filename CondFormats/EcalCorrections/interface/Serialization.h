#ifndef CondFormats_EcalCorrections_Serialization_H
#define CondFormats_EcalCorrections_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void EcalGlobalShowerContainmentCorrectionsVsEta::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(coefficients_);
}

template <class Archive>
void EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}

template <class Archive>
void EcalShowerContainmentCorrections::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(groupmap_);
    ar & BOOST_SERIALIZATION_NVP(coefficients_);
}

template <class Archive>
void EcalShowerContainmentCorrections::Coefficients::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}

namespace cond {
namespace serialization {

template <>
struct access<EcalGlobalShowerContainmentCorrectionsVsEta>
{
    static bool equal_(const EcalGlobalShowerContainmentCorrectionsVsEta & first, const EcalGlobalShowerContainmentCorrectionsVsEta & second)
    {
        return true
            and (equal(first.coefficients_, second.coefficients_))
        ;
    }
};

template <>
struct access<EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients>
{
    static bool equal_(const EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients & first, const EcalGlobalShowerContainmentCorrectionsVsEta::Coefficients & second)
    {
        return true
            and (equal(first.data, second.data))
        ;
    }
};

template <>
struct access<EcalShowerContainmentCorrections>
{
    static bool equal_(const EcalShowerContainmentCorrections & first, const EcalShowerContainmentCorrections & second)
    {
        return true
            and (equal(first.groupmap_, second.groupmap_))
            and (equal(first.coefficients_, second.coefficients_))
        ;
    }
};

template <>
struct access<EcalShowerContainmentCorrections::Coefficients>
{
    static bool equal_(const EcalShowerContainmentCorrections::Coefficients & first, const EcalShowerContainmentCorrections::Coefficients & second)
    {
        return true
            and (equal(first.data, second.data))
        ;
    }
};

}
}

#endif
