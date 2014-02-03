
#include "CondFormats/EgammaObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void ElectronLikelihoodCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}
COND_SERIALIZATION_INSTANTIATE(ElectronLikelihoodCalibration);

template <class Archive>
void ElectronLikelihoodCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}
COND_SERIALIZATION_INSTANTIATE(ElectronLikelihoodCalibration::Entry);

template <class Archive>
void ElectronLikelihoodCategoryData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ecaldet);
    ar & BOOST_SERIALIZATION_NVP(ptbin);
    ar & BOOST_SERIALIZATION_NVP(iclass);
    ar & BOOST_SERIALIZATION_NVP(ifullclass);
    ar & BOOST_SERIALIZATION_NVP(label);
}
COND_SERIALIZATION_INSTANTIATE(ElectronLikelihoodCategoryData);

template <class Archive>
void GBRForest::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fInitialResponse);
    ar & BOOST_SERIALIZATION_NVP(fTrees);
}
COND_SERIALIZATION_INSTANTIATE(GBRForest);

template <class Archive>
void GBRForest2D::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fInitialResponseX);
    ar & BOOST_SERIALIZATION_NVP(fInitialResponseY);
    ar & BOOST_SERIALIZATION_NVP(fTrees);
}
COND_SERIALIZATION_INSTANTIATE(GBRForest2D);

template <class Archive>
void GBRTree::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fCutIndices);
    ar & BOOST_SERIALIZATION_NVP(fCutVals);
    ar & BOOST_SERIALIZATION_NVP(fLeftIndices);
    ar & BOOST_SERIALIZATION_NVP(fRightIndices);
    ar & BOOST_SERIALIZATION_NVP(fResponses);
}
COND_SERIALIZATION_INSTANTIATE(GBRTree);

template <class Archive>
void GBRTree2D::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fCutIndices);
    ar & BOOST_SERIALIZATION_NVP(fCutVals);
    ar & BOOST_SERIALIZATION_NVP(fLeftIndices);
    ar & BOOST_SERIALIZATION_NVP(fRightIndices);
    ar & BOOST_SERIALIZATION_NVP(fResponsesX);
    ar & BOOST_SERIALIZATION_NVP(fResponsesY);
}
COND_SERIALIZATION_INSTANTIATE(GBRTree2D);

#include "CondFormats/EgammaObjects/src/SerializationManual.h"
