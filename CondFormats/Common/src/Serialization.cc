
#include "CondFormats/Common/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void ConfObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(parameters);
}
COND_SERIALIZATION_INSTANTIATE(ConfObject);

template <class Archive>
void DropBoxMetadata::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(recordSet);
}
COND_SERIALIZATION_INSTANTIATE(DropBoxMetadata);

template <class Archive>
void DropBoxMetadata::Parameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theParameters);
}
COND_SERIALIZATION_INSTANTIATE(DropBoxMetadata::Parameters);

template <class Archive>
void FileBlob::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(blob);
    ar & BOOST_SERIALIZATION_NVP(compressed);
    ar & BOOST_SERIALIZATION_NVP(isize);
}
COND_SERIALIZATION_INSTANTIATE(FileBlob);

template <class Archive>
void MultiFileBlob::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(blob);
    ar & BOOST_SERIALIZATION_NVP(positions);
    ar & BOOST_SERIALIZATION_NVP(compressed);
    ar & BOOST_SERIALIZATION_NVP(isize);
}
COND_SERIALIZATION_INSTANTIATE(MultiFileBlob);

template <class Archive>
void cond::BaseKeyed::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_key);
}
COND_SERIALIZATION_INSTANTIATE(cond::BaseKeyed);

template <class Archive>
void cond::GenericSummary::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::Summary", boost::serialization::base_object<cond::Summary>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_me);
}
COND_SERIALIZATION_INSTANTIATE(cond::GenericSummary);

template <class Archive>
void cond::IOVDescription::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(cond::IOVDescription);

template <class Archive>
void cond::IOVKeysDescription::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::IOVDescription", boost::serialization::base_object<cond::IOVDescription>(*this));
    ar & BOOST_SERIALIZATION_NVP(dict_m);
    ar & BOOST_SERIALIZATION_NVP(m_tag);
}
COND_SERIALIZATION_INSTANTIATE(cond::IOVKeysDescription);

template <class Archive>
void cond::IOVProvenance::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(cond::IOVProvenance);

template <class Archive>
void cond::IOVUserMetaData::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(cond::IOVUserMetaData);

template <class Archive>
void cond::SmallWORMDict::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_data);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}
COND_SERIALIZATION_INSTANTIATE(cond::SmallWORMDict);

template <class Archive>
void cond::Summary::serialize(Archive & ar, const unsigned int)
{
}
COND_SERIALIZATION_INSTANTIATE(cond::Summary);

template <class Archive>
void cond::UpdateStamp::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_revision);
    ar & BOOST_SERIALIZATION_NVP(m_timestamp);
    ar & BOOST_SERIALIZATION_NVP(m_comment);
}
COND_SERIALIZATION_INSTANTIATE(cond::UpdateStamp);

#include "CondFormats/Common/src/SerializationManual.h"
