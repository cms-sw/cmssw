#ifndef CondFormats_RecoMuonObjects_Serialization_H
#define CondFormats_RecoMuonObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void MuScleFitDBobject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(identifiers);
    ar & BOOST_SERIALIZATION_NVP(parameters);
    ar & BOOST_SERIALIZATION_NVP(fitQuality);
}

namespace cond {
namespace serialization {

template <>
struct access<MuScleFitDBobject>
{
    static bool equal_(const MuScleFitDBobject & first, const MuScleFitDBobject & second)
    {
        return true
            and (equal(first.identifiers, second.identifiers))
            and (equal(first.parameters, second.parameters))
            and (equal(first.fitQuality, second.fitQuality))
        ;
    }
};

}
}

#endif
