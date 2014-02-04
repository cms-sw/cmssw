
#include "CondFormats/Alignment/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void AlignTransform::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_translation);
    ar & BOOST_SERIALIZATION_NVP(m_eulerAngles);
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
}
COND_SERIALIZATION_INSTANTIATE(AlignTransform);

template <class Archive>
void AlignTransformError::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_Parameters);
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
}
COND_SERIALIZATION_INSTANTIATE(AlignTransformError);

template <class Archive>
void AlignmentErrors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_alignError);
}
COND_SERIALIZATION_INSTANTIATE(AlignmentErrors);

template <class Archive>
void AlignmentSurfaceDeformations::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_parameters);
    ar & BOOST_SERIALIZATION_NVP(m_items);
}
COND_SERIALIZATION_INSTANTIATE(AlignmentSurfaceDeformations);

template <class Archive>
void AlignmentSurfaceDeformations::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
    ar & BOOST_SERIALIZATION_NVP(m_parametrizationType);
    ar & BOOST_SERIALIZATION_NVP(m_index);
}
COND_SERIALIZATION_INSTANTIATE(AlignmentSurfaceDeformations::Item);

template <class Archive>
void Alignments::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_align);
}
COND_SERIALIZATION_INSTANTIATE(Alignments);

template <class Archive>
void SurveyError::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_structureType);
    ar & BOOST_SERIALIZATION_NVP(m_rawId);
    ar & BOOST_SERIALIZATION_NVP(m_errors);
}
COND_SERIALIZATION_INSTANTIATE(SurveyError);

template <class Archive>
void SurveyErrors::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_surveyErrors);
}
COND_SERIALIZATION_INSTANTIATE(SurveyErrors);

#include "CondFormats/Alignment/src/SerializationManual.h"
