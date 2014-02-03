
#include "CondFormats/BTauObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void CombinedSVCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}
COND_SERIALIZATION_INSTANTIATE(CombinedSVCalibration);

template <class Archive>
void CombinedSVCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}
COND_SERIALIZATION_INSTANTIATE(CombinedSVCalibration::Entry);

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
COND_SERIALIZATION_INSTANTIATE(CombinedSVCategoryData);

template <class Archive>
void CombinedTauTagCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}
COND_SERIALIZATION_INSTANTIATE(CombinedTauTagCalibration);

template <class Archive>
void CombinedTauTagCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}
COND_SERIALIZATION_INSTANTIATE(CombinedTauTagCalibration::Entry);

template <class Archive>
void CombinedTauTagCategoryData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(truthmatched1orfake0candidates);
    ar & BOOST_SERIALIZATION_NVP(theTagVar);
    ar & BOOST_SERIALIZATION_NVP(signaltks_n);
    ar & BOOST_SERIALIZATION_NVP(EtMin);
    ar & BOOST_SERIALIZATION_NVP(EtMax);
}
COND_SERIALIZATION_INSTANTIATE(CombinedTauTagCategoryData);

template <class Archive>
void TrackProbabilityCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}
COND_SERIALIZATION_INSTANTIATE(TrackProbabilityCalibration);

template <class Archive>
void TrackProbabilityCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}
COND_SERIALIZATION_INSTANTIATE(TrackProbabilityCalibration::Entry);

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
COND_SERIALIZATION_INSTANTIATE(TrackProbabilityCategoryData);

#include "CondFormats/BTauObjects/src/SerializationManual.h"
