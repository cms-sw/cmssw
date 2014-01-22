#ifndef CondFormats_EgammaObjects_Serialization_H
#define CondFormats_EgammaObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void ElectronLikelihoodCalibration::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(data);
}

template <class Archive>
void ElectronLikelihoodCalibration::Entry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(category);
    ar & BOOST_SERIALIZATION_NVP(histogram);
}

template <class Archive>
void ElectronLikelihoodCategoryData::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(ecaldet);
    ar & BOOST_SERIALIZATION_NVP(ptbin);
    ar & BOOST_SERIALIZATION_NVP(iclass);
    ar & BOOST_SERIALIZATION_NVP(ifullclass);
    ar & BOOST_SERIALIZATION_NVP(label);
}

template <class Archive>
void GBRForest::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fInitialResponse);
    ar & BOOST_SERIALIZATION_NVP(fTrees);
}

template <class Archive>
void GBRForest2D::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fInitialResponseX);
    ar & BOOST_SERIALIZATION_NVP(fInitialResponseY);
    ar & BOOST_SERIALIZATION_NVP(fTrees);
}

template <class Archive>
void GBRTree::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fCutIndices);
    ar & BOOST_SERIALIZATION_NVP(fCutVals);
    ar & BOOST_SERIALIZATION_NVP(fLeftIndices);
    ar & BOOST_SERIALIZATION_NVP(fRightIndices);
    ar & BOOST_SERIALIZATION_NVP(fResponses);
}

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

namespace cond {
namespace serialization {

template <>
struct access<ElectronLikelihoodCalibration>
{
    static bool equal_(const ElectronLikelihoodCalibration & first, const ElectronLikelihoodCalibration & second)
    {
        return true
            and (equal(first.data, second.data))
        ;
    }
};

template <>
struct access<ElectronLikelihoodCalibration::Entry>
{
    static bool equal_(const ElectronLikelihoodCalibration::Entry & first, const ElectronLikelihoodCalibration::Entry & second)
    {
        return true
            and (equal(first.category, second.category))
            and (equal(first.histogram, second.histogram))
        ;
    }
};

template <>
struct access<ElectronLikelihoodCategoryData>
{
    static bool equal_(const ElectronLikelihoodCategoryData & first, const ElectronLikelihoodCategoryData & second)
    {
        return true
            and (equal(first.ecaldet, second.ecaldet))
            and (equal(first.ptbin, second.ptbin))
            and (equal(first.iclass, second.iclass))
            and (equal(first.ifullclass, second.ifullclass))
            and (equal(first.label, second.label))
        ;
    }
};

template <>
struct access<GBRForest>
{
    static bool equal_(const GBRForest & first, const GBRForest & second)
    {
        return true
            and (equal(first.fInitialResponse, second.fInitialResponse))
            and (equal(first.fTrees, second.fTrees))
        ;
    }
};

template <>
struct access<GBRForest2D>
{
    static bool equal_(const GBRForest2D & first, const GBRForest2D & second)
    {
        return true
            and (equal(first.fInitialResponseX, second.fInitialResponseX))
            and (equal(first.fInitialResponseY, second.fInitialResponseY))
            and (equal(first.fTrees, second.fTrees))
        ;
    }
};

template <>
struct access<GBRTree>
{
    static bool equal_(const GBRTree & first, const GBRTree & second)
    {
        return true
            and (equal(first.fCutIndices, second.fCutIndices))
            and (equal(first.fCutVals, second.fCutVals))
            and (equal(first.fLeftIndices, second.fLeftIndices))
            and (equal(first.fRightIndices, second.fRightIndices))
            and (equal(first.fResponses, second.fResponses))
        ;
    }
};

template <>
struct access<GBRTree2D>
{
    static bool equal_(const GBRTree2D & first, const GBRTree2D & second)
    {
        return true
            and (equal(first.fCutIndices, second.fCutIndices))
            and (equal(first.fCutVals, second.fCutVals))
            and (equal(first.fLeftIndices, second.fLeftIndices))
            and (equal(first.fRightIndices, second.fRightIndices))
            and (equal(first.fResponsesX, second.fResponsesX))
            and (equal(first.fResponsesY, second.fResponsesY))
        ;
    }
};

}
}

#endif
