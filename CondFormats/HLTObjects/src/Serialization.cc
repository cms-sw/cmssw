
#include "CondFormats/HLTObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void AlCaRecoTriggerBits::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_alcarecoToTrig);
}
COND_SERIALIZATION_INSTANTIATE(AlCaRecoTriggerBits);

template <class Archive>
void trigger::HLTPrescaleTableCond::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(hltPrescaleTable_);
}
COND_SERIALIZATION_INSTANTIATE(trigger::HLTPrescaleTableCond);

#include "CondFormats/HLTObjects/src/SerializationManual.h"
