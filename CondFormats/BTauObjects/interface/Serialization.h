#ifndef CondFormats_BTauObjects_Serialization_H
#define CondFormats_BTauObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void CombinedSVCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}

template <class Archive>
void CombinedSVCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}

template <class Archive>
void CombinedSVCategoryData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(JetEtaMax);
    ar & BOOST_SERIALIZATION_NVP(JetEtaMin);
    ar & BOOST_SERIALIZATION_NVP(JetPtMax);
    ar & BOOST_SERIALIZATION_NVP(JetPtMin);
    ar & BOOST_SERIALIZATION_NVP(PartonType);
    ar & BOOST_SERIALIZATION_NVP(TaggingVariable);
    ar & BOOST_SERIALIZATION_NVP(VertexType);
}

template <class Archive>
void CombinedTauTagCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}

template <class Archive>
void CombinedTauTagCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}

template <class Archive>
void CombinedTauTagCategoryData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(truthmatched1orfake0candidates);
    ar & BOOST_SERIALIZATION_NVP(theTagVar);
    ar & BOOST_SERIALIZATION_NVP(signaltks_n);
    ar & BOOST_SERIALIZATION_NVP(EtMin);
    ar & BOOST_SERIALIZATION_NVP(EtMax);
}

template <class Archive>
void TrackProbabilityCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}

template <class Archive>
void TrackProbabilityCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}

template <class Archive>
void TrackProbabilityCategoryData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pMin);
    ar & BOOST_SERIALIZATION_NVP(pMax);
    ar & BOOST_SERIALIZATION_NVP(etaMin);
    ar & BOOST_SERIALIZATION_NVP(etaMax);
    ar & BOOST_SERIALIZATION_NVP(nHitsMin);
    ar & BOOST_SERIALIZATION_NVP(nHitsMax);
    ar & BOOST_SERIALIZATION_NVP(nPixelHitsMin);
    ar & BOOST_SERIALIZATION_NVP(nPixelHitsMax);
    ar & BOOST_SERIALIZATION_NVP(chiMin);
    ar & BOOST_SERIALIZATION_NVP(chiMax);
    ar & BOOST_SERIALIZATION_NVP(withFirstPixel);
    ar & BOOST_SERIALIZATION_NVP(trackQuality);
}

namespace cond {
namespace serialization {

template <>
struct access<CombinedSVCalibration>
{
    static bool equal_(const CombinedSVCalibration & first, const CombinedSVCalibration & second)
    {
        return true
            and (equal(first.data, second.data))
        ;
    }
};

template <>
struct access<CombinedSVCalibration::Entry>
{
    static bool equal_(const CombinedSVCalibration::Entry & first, const CombinedSVCalibration::Entry & second)
    {
        return true
            and (equal(first.category, second.category))
            and (equal(first.histogram, second.histogram))
        ;
    }
};

template <>
struct access<CombinedSVCategoryData>
{
    static bool equal_(const CombinedSVCategoryData & first, const CombinedSVCategoryData & second)
    {
        return true
            and (equal(first.JetEtaMax, second.JetEtaMax))
            and (equal(first.JetEtaMin, second.JetEtaMin))
            and (equal(first.JetPtMax, second.JetPtMax))
            and (equal(first.JetPtMin, second.JetPtMin))
            and (equal(first.PartonType, second.PartonType))
            and (equal(first.TaggingVariable, second.TaggingVariable))
            and (equal(first.VertexType, second.VertexType))
        ;
    }
};

template <>
struct access<CombinedTauTagCalibration>
{
    static bool equal_(const CombinedTauTagCalibration & first, const CombinedTauTagCalibration & second)
    {
        return true
            and (equal(first.data, second.data))
        ;
    }
};

template <>
struct access<CombinedTauTagCalibration::Entry>
{
    static bool equal_(const CombinedTauTagCalibration::Entry & first, const CombinedTauTagCalibration::Entry & second)
    {
        return true
            and (equal(first.category, second.category))
            and (equal(first.histogram, second.histogram))
        ;
    }
};

template <>
struct access<CombinedTauTagCategoryData>
{
    static bool equal_(const CombinedTauTagCategoryData & first, const CombinedTauTagCategoryData & second)
    {
        return true
            and (equal(first.truthmatched1orfake0candidates, second.truthmatched1orfake0candidates))
            and (equal(first.theTagVar, second.theTagVar))
            and (equal(first.signaltks_n, second.signaltks_n))
            and (equal(first.EtMin, second.EtMin))
            and (equal(first.EtMax, second.EtMax))
        ;
    }
};

template <>
struct access<TrackProbabilityCalibration>
{
    static bool equal_(const TrackProbabilityCalibration & first, const TrackProbabilityCalibration & second)
    {
        return true
            and (equal(first.data, second.data))
        ;
    }
};

template <>
struct access<TrackProbabilityCalibration::Entry>
{
    static bool equal_(const TrackProbabilityCalibration::Entry & first, const TrackProbabilityCalibration::Entry & second)
    {
        return true
            and (equal(first.category, second.category))
            and (equal(first.histogram, second.histogram))
        ;
    }
};

template <>
struct access<TrackProbabilityCategoryData>
{
    static bool equal_(const TrackProbabilityCategoryData & first, const TrackProbabilityCategoryData & second)
    {
        return true
            and (equal(first.pMin, second.pMin))
            and (equal(first.pMax, second.pMax))
            and (equal(first.etaMin, second.etaMin))
            and (equal(first.etaMax, second.etaMax))
            and (equal(first.nHitsMin, second.nHitsMin))
            and (equal(first.nHitsMax, second.nHitsMax))
            and (equal(first.nPixelHitsMin, second.nPixelHitsMin))
            and (equal(first.nPixelHitsMax, second.nPixelHitsMax))
            and (equal(first.chiMin, second.chiMin))
            and (equal(first.chiMax, second.chiMax))
            and (equal(first.withFirstPixel, second.withFirstPixel))
            and (equal(first.trackQuality, second.trackQuality))
        ;
    }
};

}
}

#endif
