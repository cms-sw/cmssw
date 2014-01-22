#ifndef CondFormats_Luminosity_Serialization_H
#define CondFormats_Luminosity_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void lumi::BunchCrossingInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(BXIdx);
    ar & BOOST_SERIALIZATION_NVP(lumivalue);
    ar & BOOST_SERIALIZATION_NVP(lumierr);
    ar & BOOST_SERIALIZATION_NVP(lumiquality);
}

template <class Archive>
void lumi::HLTInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pathname);
    ar & BOOST_SERIALIZATION_NVP(inputcount);
    ar & BOOST_SERIALIZATION_NVP(acceptcount);
    ar & BOOST_SERIALIZATION_NVP(prescale);
}

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

template <class Archive>
void lumi::TriggerInfo::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(name);
    ar & BOOST_SERIALIZATION_NVP(triggercount);
    ar & BOOST_SERIALIZATION_NVP(deadtimecount);
    ar & BOOST_SERIALIZATION_NVP(prescale);
}

namespace cond {
namespace serialization {

template <>
struct access<lumi::BunchCrossingInfo>
{
    static bool equal_(const lumi::BunchCrossingInfo & first, const lumi::BunchCrossingInfo & second)
    {
        return true
            and (equal(first.BXIdx, second.BXIdx))
            and (equal(first.lumivalue, second.lumivalue))
            and (equal(first.lumierr, second.lumierr))
            and (equal(first.lumiquality, second.lumiquality))
        ;
    }
};

template <>
struct access<lumi::HLTInfo>
{
    static bool equal_(const lumi::HLTInfo & first, const lumi::HLTInfo & second)
    {
        return true
            and (equal(first.pathname, second.pathname))
            and (equal(first.inputcount, second.inputcount))
            and (equal(first.acceptcount, second.acceptcount))
            and (equal(first.prescale, second.prescale))
        ;
    }
};

template <>
struct access<lumi::LumiSectionData>
{
    static bool equal_(const lumi::LumiSectionData & first, const lumi::LumiSectionData & second)
    {
        return true
            and (equal(first.m_bx, second.m_bx))
            and (equal(first.m_sectionid, second.m_sectionid))
            and (equal(first.m_versionid, second.m_versionid))
            and (equal(first.m_lumiavg, second.m_lumiavg))
            and (equal(first.m_lumierror, second.m_lumierror))
            and (equal(first.m_quality, second.m_quality))
            and (equal(first.m_deadfrac, second.m_deadfrac))
            and (equal(first.m_startorbit, second.m_startorbit))
            and (equal(first.m_hlt, second.m_hlt))
            and (equal(first.m_trigger, second.m_trigger))
        ;
    }
};

template <>
struct access<lumi::TriggerInfo>
{
    static bool equal_(const lumi::TriggerInfo & first, const lumi::TriggerInfo & second)
    {
        return true
            and (equal(first.name, second.name))
            and (equal(first.triggercount, second.triggercount))
            and (equal(first.deadtimecount, second.deadtimecount))
            and (equal(first.prescale, second.prescale))
        ;
    }
};

}
}

#endif
