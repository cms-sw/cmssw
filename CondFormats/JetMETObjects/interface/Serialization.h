#ifndef CondFormats_JetMETObjects_Serialization_H
#define CondFormats_JetMETObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void FFTJetCorrectorParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_buffer);
}

template <class Archive>
void JetCorrectorParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mDefinitions);
    ar & BOOST_SERIALIZATION_NVP(mRecords);
    ar & BOOST_SERIALIZATION_NVP(valid_);
}

template <class Archive>
void JetCorrectorParameters::Definitions::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mIsResponse);
    ar & BOOST_SERIALIZATION_NVP(mLevel);
    ar & BOOST_SERIALIZATION_NVP(mFormula);
    ar & BOOST_SERIALIZATION_NVP(mParVar);
    ar & BOOST_SERIALIZATION_NVP(mBinVar);
}

template <class Archive>
void JetCorrectorParameters::Record::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(mNvar);
    ar & BOOST_SERIALIZATION_NVP(mMin);
    ar & BOOST_SERIALIZATION_NVP(mMax);
    ar & BOOST_SERIALIZATION_NVP(mParameters);
}

template <class Archive>
void JetCorrectorParametersCollection::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(corrections_);
    ar & BOOST_SERIALIZATION_NVP(correctionsL5_);
    ar & BOOST_SERIALIZATION_NVP(correctionsL7_);
}

namespace cond {
namespace serialization {

template <>
struct access<FFTJetCorrectorParameters>
{
    static bool equal_(const FFTJetCorrectorParameters & first, const FFTJetCorrectorParameters & second)
    {
        return true
            and (equal(first.m_buffer, second.m_buffer))
        ;
    }
};

template <>
struct access<JetCorrectorParameters>
{
    static bool equal_(const JetCorrectorParameters & first, const JetCorrectorParameters & second)
    {
        return true
            and (equal(first.mDefinitions, second.mDefinitions))
            and (equal(first.mRecords, second.mRecords))
            and (equal(first.valid_, second.valid_))
        ;
    }
};

template <>
struct access<JetCorrectorParameters::Definitions>
{
    static bool equal_(const JetCorrectorParameters::Definitions & first, const JetCorrectorParameters::Definitions & second)
    {
        return true
            and (equal(first.mIsResponse, second.mIsResponse))
            and (equal(first.mLevel, second.mLevel))
            and (equal(first.mFormula, second.mFormula))
            and (equal(first.mParVar, second.mParVar))
            and (equal(first.mBinVar, second.mBinVar))
        ;
    }
};

template <>
struct access<JetCorrectorParameters::Record>
{
    static bool equal_(const JetCorrectorParameters::Record & first, const JetCorrectorParameters::Record & second)
    {
        return true
            and (equal(first.mNvar, second.mNvar))
            and (equal(first.mMin, second.mMin))
            and (equal(first.mMax, second.mMax))
            and (equal(first.mParameters, second.mParameters))
        ;
    }
};

template <>
struct access<JetCorrectorParametersCollection>
{
    static bool equal_(const JetCorrectorParametersCollection & first, const JetCorrectorParametersCollection & second)
    {
        return true
            and (equal(first.corrections_, second.corrections_))
            and (equal(first.correctionsL5_, second.correctionsL5_))
            and (equal(first.correctionsL7_, second.correctionsL7_))
        ;
    }
};

}
}

#endif
