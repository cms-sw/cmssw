
#include "CondFormats/RecoMuonObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void MuScleFitDBobject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(identifiers);
    ar & BOOST_SERIALIZATION_NVP(parameters);
    ar & BOOST_SERIALIZATION_NVP(fitQuality);
}
COND_SERIALIZATION_INSTANTIATE(MuScleFitDBobject);

#include "CondFormats/RecoMuonObjects/src/SerializationManual.h"
