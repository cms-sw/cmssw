#ifndef CondFormats_HLTObjects_Serialization_H
#define CondFormats_HLTObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void AlCaRecoTriggerBits::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_alcarecoToTrig);
}

namespace cond {
namespace serialization {

template <>
struct access<AlCaRecoTriggerBits>
{
    static bool equal_(const AlCaRecoTriggerBits & first, const AlCaRecoTriggerBits & second)
    {
        return true
            and (equal(first.m_alcarecoToTrig, second.m_alcarecoToTrig))
        ;
    }
};

}
}

#endif
