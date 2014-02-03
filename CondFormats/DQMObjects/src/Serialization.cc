
#include "CondFormats/DQMObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void DQMSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_run);
    ar & BOOST_SERIALIZATION_NVP(m_summary);
}
COND_SERIALIZATION_INSTANTIATE(DQMSummary);

template <class Archive>
void DQMSummary::RunItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_lumisec);
    ar & BOOST_SERIALIZATION_NVP(m_lumisummary);
}
COND_SERIALIZATION_INSTANTIATE(DQMSummary::RunItem);

template <class Archive>
void DQMSummary::RunItem::LumiItem::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_subsystem);
    ar & BOOST_SERIALIZATION_NVP(m_reportcontent);
    ar & BOOST_SERIALIZATION_NVP(m_type);
    ar & BOOST_SERIALIZATION_NVP(m_status);
}
COND_SERIALIZATION_INSTANTIATE(DQMSummary::RunItem::LumiItem);

template <class Archive>
void HDQMSummary::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(userDBContent_);
    ar & BOOST_SERIALIZATION_NVP(v_sum_);
    ar & BOOST_SERIALIZATION_NVP(indexes_);
    ar & BOOST_SERIALIZATION_NVP(runNr_);
    ar & BOOST_SERIALIZATION_NVP(timeValue_);
}
COND_SERIALIZATION_INSTANTIATE(HDQMSummary);

template <class Archive>
void HDQMSummary::DetRegistry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(detid);
    ar & BOOST_SERIALIZATION_NVP(ibegin);
}
COND_SERIALIZATION_INSTANTIATE(HDQMSummary::DetRegistry);

#include "CondFormats/DQMObjects/src/SerializationManual.h"
