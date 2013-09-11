#ifndef CondFormats_Common_Serialization_H
#define CondFormats_Common_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/Serialization/interface/SerializationEqual.h"

// #include "CondFormats/External/interface/Serialization.h"

#include "CondFormats/Common/src/headers.h"

template <class Archive>
void ConfObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(parameters);
}

template <class Archive>
void DropBoxMetadata::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(recordSet);
}

template <class Archive>
void DropBoxMetadata::Parameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(theParameters);
}

template <class Archive>
void FileBlob::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(blob);
    ar & BOOST_SERIALIZATION_NVP(compressed);
    ar & BOOST_SERIALIZATION_NVP(isize);
}

template <class Archive>
void MultiFileBlob::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(blob);
    ar & BOOST_SERIALIZATION_NVP(positions);
    ar & BOOST_SERIALIZATION_NVP(compressed);
    ar & BOOST_SERIALIZATION_NVP(isize);
}

template <class Archive>
void cond::BaseKeyed::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_key);
}

template <class Archive>
void cond::GenericSummary::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::Summary", boost::serialization::base_object<cond::Summary>(*this));
    ar & BOOST_SERIALIZATION_NVP(m_me);
}

template <class Archive>
void cond::IOVDescription::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void cond::IOVKeysDescription::serialize(Archive & ar, const unsigned int)
{
    ar & boost::serialization::make_nvp("cond::IOVDescription", boost::serialization::base_object<cond::IOVDescription>(*this));
    ar & BOOST_SERIALIZATION_NVP(dict_m);
    ar & BOOST_SERIALIZATION_NVP(m_tag);
}

template <class Archive>
void cond::IOVProvenance::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void cond::IOVUserMetaData::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void cond::SmallWORMDict::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_data);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}

template <class Archive>
void cond::Summary::serialize(Archive & ar, const unsigned int)
{
}

template <class Archive>
void cond::UpdateStamp::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_revision);
    ar & BOOST_SERIALIZATION_NVP(m_timestamp);
    ar & BOOST_SERIALIZATION_NVP(m_comment);
}

namespace cond {
namespace serialization {

template <>
struct access<ConfObject>
{
    static bool equal_(const ConfObject & first, const ConfObject & second)
    {
        return true
            and (equal(first.parameters, second.parameters))
        ;
    }
};

template <>
struct access<DropBoxMetadata>
{
    static bool equal_(const DropBoxMetadata & first, const DropBoxMetadata & second)
    {
        return true
            and (equal(first.recordSet, second.recordSet))
        ;
    }
};

template <>
struct access<DropBoxMetadata::Parameters>
{
    static bool equal_(const DropBoxMetadata::Parameters & first, const DropBoxMetadata::Parameters & second)
    {
        return true
            and (equal(first.theParameters, second.theParameters))
        ;
    }
};

template <>
struct access<FileBlob>
{
    static bool equal_(const FileBlob & first, const FileBlob & second)
    {
        return true
            and (equal(first.blob, second.blob))
            and (equal(first.compressed, second.compressed))
            and (equal(first.isize, second.isize))
        ;
    }
};

template <>
struct access<MultiFileBlob>
{
    static bool equal_(const MultiFileBlob & first, const MultiFileBlob & second)
    {
        return true
            and (equal(first.blob, second.blob))
            and (equal(first.positions, second.positions))
            and (equal(first.compressed, second.compressed))
            and (equal(first.isize, second.isize))
        ;
    }
};

template <>
struct access<cond::BaseKeyed>
{
    static bool equal_(const cond::BaseKeyed & first, const cond::BaseKeyed & second)
    {
        return true
            and (equal(first.m_key, second.m_key))
        ;
    }
};

template <>
struct access<cond::GenericSummary>
{
    static bool equal_(const cond::GenericSummary & first, const cond::GenericSummary & second)
    {
        return true
            and (equal(static_cast<const cond::Summary &>(first), static_cast<const cond::Summary &>(second)))
            and (equal(first.m_me, second.m_me))
        ;
    }
};

template <>
struct access<cond::IOVDescription>
{
    static bool equal_(const cond::IOVDescription & first, const cond::IOVDescription & second)
    {
        return true
        ;
    }
};

template <>
struct access<cond::IOVKeysDescription>
{
    static bool equal_(const cond::IOVKeysDescription & first, const cond::IOVKeysDescription & second)
    {
        return true
            and (equal(static_cast<const cond::IOVDescription &>(first), static_cast<const cond::IOVDescription &>(second)))
            and (equal(first.dict_m, second.dict_m))
            and (equal(first.m_tag, second.m_tag))
        ;
    }
};

template <>
struct access<cond::IOVProvenance>
{
    static bool equal_(const cond::IOVProvenance & first, const cond::IOVProvenance & second)
    {
        return true
        ;
    }
};

template <>
struct access<cond::IOVUserMetaData>
{
    static bool equal_(const cond::IOVUserMetaData & first, const cond::IOVUserMetaData & second)
    {
        return true
        ;
    }
};

template <>
struct access<cond::SmallWORMDict>
{
    static bool equal_(const cond::SmallWORMDict & first, const cond::SmallWORMDict & second)
    {
        return true
            and (equal(first.m_data, second.m_data))
            and (equal(first.m_index, second.m_index))
        ;
    }
};

template <>
struct access<cond::Summary>
{
    static bool equal_(const cond::Summary & first, const cond::Summary & second)
    {
        return true
        ;
    }
};

template <>
struct access<cond::UpdateStamp>
{
    static bool equal_(const cond::UpdateStamp & first, const cond::UpdateStamp & second)
    {
        return true
            and (equal(first.m_revision, second.m_revision))
            and (equal(first.m_timestamp, second.m_timestamp))
            and (equal(first.m_comment, second.m_comment))
        ;
    }
};

}
}

#endif
