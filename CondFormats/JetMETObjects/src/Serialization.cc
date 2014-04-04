
#include "CondFormats/JetMETObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void FFTJetCorrectorParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_buffer);
}
COND_SERIALIZATION_INSTANTIATE(FFTJetCorrectorParameters);

template <class Archive>
void JetCorrectorParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mDefinitions);
    ar & BOOST_SERIALIZATION_NVP(mRecords);
    ar & BOOST_SERIALIZATION_NVP(valid_);
}
COND_SERIALIZATION_INSTANTIATE(JetCorrectorParameters);

template <class Archive>
void JetCorrectorParameters::Definitions::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mIsResponse);
    ar & BOOST_SERIALIZATION_NVP(mLevel);
    ar & BOOST_SERIALIZATION_NVP(mFormula);
    ar & BOOST_SERIALIZATION_NVP(mParVar);
    ar & BOOST_SERIALIZATION_NVP(mBinVar);
}
COND_SERIALIZATION_INSTANTIATE(JetCorrectorParameters::Definitions);

template <class Archive>
void JetCorrectorParameters::Record::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mNvar);
    ar & BOOST_SERIALIZATION_NVP(mMin);
    ar & BOOST_SERIALIZATION_NVP(mMax);
    ar & BOOST_SERIALIZATION_NVP(mParameters);
}
COND_SERIALIZATION_INSTANTIATE(JetCorrectorParameters::Record);

template <class Archive>
void JetCorrectorParametersCollection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(corrections_);
    ar & BOOST_SERIALIZATION_NVP(correctionsL5_);
    ar & BOOST_SERIALIZATION_NVP(correctionsL7_);
}
COND_SERIALIZATION_INSTANTIATE(JetCorrectorParametersCollection);

template <class Archive>
void QGLikelihoodCategory::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(RhoVal);
    ar & BOOST_SERIALIZATION_NVP(PtMin);
    ar & BOOST_SERIALIZATION_NVP(PtMax);
    ar & BOOST_SERIALIZATION_NVP(EtaBin);
    ar & BOOST_SERIALIZATION_NVP(QGIndex);
    ar & BOOST_SERIALIZATION_NVP(VarIndex);
}
COND_SERIALIZATION_INSTANTIATE(QGLikelihoodCategory);

template <class Archive>
void QGLikelihoodObject::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}
COND_SERIALIZATION_INSTANTIATE(QGLikelihoodObject);

template <class Archive>
void QGLikelihoodObject::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}
COND_SERIALIZATION_INSTANTIATE(QGLikelihoodObject::Entry);

#include "CondFormats/JetMETObjects/src/SerializationManual.h"
