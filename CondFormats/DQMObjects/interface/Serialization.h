#ifndef CondFormats_DQMObjects_Serialization_H
#define CondFormats_DQMObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void DQMSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_run);
    ar & BOOST_SERIALIZATION_NVP(m_summary);
}

template <class Archive>
void DQMSummary::RunItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_lumisec);
    ar & BOOST_SERIALIZATION_NVP(m_lumisummary);
}

template <class Archive>
void DQMSummary::RunItem::LumiItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_subsystem);
    ar & BOOST_SERIALIZATION_NVP(m_reportcontent);
    ar & BOOST_SERIALIZATION_NVP(m_type);
    ar & BOOST_SERIALIZATION_NVP(m_status);
}

template <class Archive>
void HDQMSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(userDBContent_);
    ar & BOOST_SERIALIZATION_NVP(v_sum_);
    ar & BOOST_SERIALIZATION_NVP(indexes_);
    ar & BOOST_SERIALIZATION_NVP(runNr_);
    ar & BOOST_SERIALIZATION_NVP(timeValue_);
}

template <class Archive>
void HDQMSummary::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
}

namespace cond {
namespace serialization {

template <>
struct access<DQMSummary>
{
    static bool equal_(const DQMSummary & first, const DQMSummary & second)
    {
        return true
            and (equal(first.m_run, second.m_run))
            and (equal(first.m_summary, second.m_summary))
        ;
    }
};

template <>
struct access<DQMSummary::RunItem>
{
    static bool equal_(const DQMSummary::RunItem & first, const DQMSummary::RunItem & second)
    {
        return true
            and (equal(first.m_lumisec, second.m_lumisec))
            and (equal(first.m_lumisummary, second.m_lumisummary))
        ;
    }
};

template <>
struct access<DQMSummary::RunItem::LumiItem>
{
    static bool equal_(const DQMSummary::RunItem::LumiItem & first, const DQMSummary::RunItem::LumiItem & second)
    {
        return true
            and (equal(first.m_subsystem, second.m_subsystem))
            and (equal(first.m_reportcontent, second.m_reportcontent))
            and (equal(first.m_type, second.m_type))
            and (equal(first.m_status, second.m_status))
        ;
    }
};

template <>
struct access<HDQMSummary>
{
    static bool equal_(const HDQMSummary & first, const HDQMSummary & second)
    {
        return true
            and (equal(first.userDBContent_, second.userDBContent_))
            and (equal(first.v_sum_, second.v_sum_))
            and (equal(first.indexes_, second.indexes_))
            and (equal(first.runNr_, second.runNr_))
            and (equal(first.timeValue_, second.timeValue_))
        ;
    }
};

template <>
struct access<HDQMSummary::DetRegistry>
{
    static bool equal_(const HDQMSummary::DetRegistry & first, const HDQMSummary::DetRegistry & second)
    {
        return true
            and (equal(first.detid, second.detid))
            and (equal(first.ibegin, second.ibegin))
        ;
    }
};

}
}

#endif
