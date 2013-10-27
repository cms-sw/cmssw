#ifndef CondFormats_Alignment_Serialization_H
#define CondFormats_Alignment_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void AlignTransform::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_translation);
    ar & BOOST_SERIALIZATION_NVP(m_eulerAngles);
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
}

template <class Archive>
void AlignTransformError::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_Parameters);
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
}

template <class Archive>
void AlignmentErrors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_alignError);
}

template <class Archive>
void AlignmentSurfaceDeformations::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_parameters);
    ar & BOOST_SERIALIZATION_NVP(m_items);
}

template <class Archive>
void AlignmentSurfaceDeformations::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
    ar & BOOST_SERIALIZATION_NVP(m_parametrizationType);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}

template <class Archive>
void Alignments::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_align);
}

template <class Archive>
void SurveyError::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_structureType);
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
    ar & BOOST_SERIALIZATION_NVP(m_errors);
}

template <class Archive>
void SurveyErrors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_surveyErrors);
}

namespace cond {
namespace serialization {

template <>
struct access<AlignTransform>
{
    static bool equal_(const AlignTransform & first, const AlignTransform & second)
    {
        return true
            and (equal(first.m_translation, second.m_translation))
            and (equal(first.m_eulerAngles, second.m_eulerAngles))
            and (equal(first.m_rawId, second.m_rawId))
        ;
    }
};

template <>
struct access<AlignTransformError>
{
    static bool equal_(const AlignTransformError & first, const AlignTransformError & second)
    {
        return true
            and (equal(first.m_Parameters, second.m_Parameters))
            and (equal(first.m_rawId, second.m_rawId))
        ;
    }
};

template <>
struct access<AlignmentErrors>
{
    static bool equal_(const AlignmentErrors & first, const AlignmentErrors & second)
    {
        return true
            and (equal(first.m_alignError, second.m_alignError))
        ;
    }
};

template <>
struct access<AlignmentSurfaceDeformations>
{
    static bool equal_(const AlignmentSurfaceDeformations & first, const AlignmentSurfaceDeformations & second)
    {
        return true
            and (equal(first.m_parameters, second.m_parameters))
            and (equal(first.m_items, second.m_items))
        ;
    }
};

template <>
struct access<AlignmentSurfaceDeformations::Item>
{
    static bool equal_(const AlignmentSurfaceDeformations::Item & first, const AlignmentSurfaceDeformations::Item & second)
    {
        return true
            and (equal(first.m_rawId, second.m_rawId))
            and (equal(first.m_parametrizationType, second.m_parametrizationType))
            and (equal(first.m_index, second.m_index))
        ;
    }
};

template <>
struct access<Alignments>
{
    static bool equal_(const Alignments & first, const Alignments & second)
    {
        return true
            and (equal(first.m_align, second.m_align))
        ;
    }
};

template <>
struct access<SurveyError>
{
    static bool equal_(const SurveyError & first, const SurveyError & second)
    {
        return true
            and (equal(first.m_structureType, second.m_structureType))
            and (equal(first.m_rawId, second.m_rawId))
            and (equal(first.m_errors, second.m_errors))
        ;
    }
};

template <>
struct access<SurveyErrors>
{
    static bool equal_(const SurveyErrors & first, const SurveyErrors & second)
    {
        return true
            and (equal(first.m_surveyErrors, second.m_surveyErrors))
        ;
    }
};

}
}

#endif
