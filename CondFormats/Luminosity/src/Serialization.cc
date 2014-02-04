
#include "CondFormats/Luminosity/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void lumi::BunchCrossingInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(BXIdx);
    ar & BOOST_SERIALIZATION_NVP(lumivalue);
    ar & BOOST_SERIALIZATION_NVP(lumierr);
    ar & BOOST_SERIALIZATION_NVP(lumiquality);
}
COND_SERIALIZATION_INSTANTIATE(lumi::BunchCrossingInfo);

template <class Archive>
void lumi::HLTInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pathname);
    ar & BOOST_SERIALIZATION_NVP(inputcount);
    ar & BOOST_SERIALIZATION_NVP(acceptcount);
    ar & BOOST_SERIALIZATION_NVP(prescale);
}
COND_SERIALIZATION_INSTANTIATE(lumi::HLTInfo);

template <class Archive>
void lumi::LumiSectionData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_bx);
    ar & BOOST_SERIALIZATION_NVP(m_sectionid);
    ar & BOOST_SERIALIZATION_NVP(m_versionid);
    ar & BOOST_SERIALIZATION_NVP(m_lumiavg);
    ar & BOOST_SERIALIZATION_NVP(m_lumierror);
    ar & BOOST_SERIALIZATION_NVP(m_quality);
    ar & BOOST_SERIALIZATION_NVP(m_deadfrac);
    ar & BOOST_SERIALIZATION_NVP(m_startorbit);
    ar & BOOST_SERIALIZATION_NVP(m_hlt);
    ar & BOOST_SERIALIZATION_NVP(m_trigger);
}
COND_SERIALIZATION_INSTANTIATE(lumi::LumiSectionData);

template <class Archive>
void lumi::TriggerInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(name);
    ar & BOOST_SERIALIZATION_NVP(triggercount);
    ar & BOOST_SERIALIZATION_NVP(deadtimecount);
    ar & BOOST_SERIALIZATION_NVP(prescale);
}
COND_SERIALIZATION_INSTANTIATE(lumi::TriggerInfo);

#include "CondFormats/Luminosity/src/SerializationManual.h"
