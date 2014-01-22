#ifndef CondFormats_GeometryObjects_Serialization_H
#define CondFormats_GeometryObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

// #include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void CSCRecoDigiParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pUserParOffset);
    ar & BOOST_SERIALIZATION_NVP(pUserParSize);
    ar & BOOST_SERIALIZATION_NVP(pChamberType);
    ar & BOOST_SERIALIZATION_NVP(pfupars);
}

template <class Archive>
void PCaloGeometry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_translation);
    ar & BOOST_SERIALIZATION_NVP(m_dimension);
    ar & BOOST_SERIALIZATION_NVP(m_indexes);
    ar & BOOST_SERIALIZATION_NVP(m_dins);
}

template <class Archive>
void PGeometricDet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pgeomdets_);
}

template <class Archive>
void PGeometricDet::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(_name);
    ar & BOOST_SERIALIZATION_NVP(_ns);
    ar & BOOST_SERIALIZATION_NVP(_x);
    ar & BOOST_SERIALIZATION_NVP(_y);
    ar & BOOST_SERIALIZATION_NVP(_z);
    ar & BOOST_SERIALIZATION_NVP(_phi);
    ar & BOOST_SERIALIZATION_NVP(_rho);
    ar & BOOST_SERIALIZATION_NVP(_a11);
    ar & BOOST_SERIALIZATION_NVP(_a12);
    ar & BOOST_SERIALIZATION_NVP(_a13);
    ar & BOOST_SERIALIZATION_NVP(_a21);
    ar & BOOST_SERIALIZATION_NVP(_a22);
    ar & BOOST_SERIALIZATION_NVP(_a23);
    ar & BOOST_SERIALIZATION_NVP(_a31);
    ar & BOOST_SERIALIZATION_NVP(_a32);
    ar & BOOST_SERIALIZATION_NVP(_a33);
    ar & BOOST_SERIALIZATION_NVP(_params0);
    ar & BOOST_SERIALIZATION_NVP(_params1);
    ar & BOOST_SERIALIZATION_NVP(_params2);
    ar & BOOST_SERIALIZATION_NVP(_params3);
    ar & BOOST_SERIALIZATION_NVP(_params4);
    ar & BOOST_SERIALIZATION_NVP(_params5);
    ar & BOOST_SERIALIZATION_NVP(_params6);
    ar & BOOST_SERIALIZATION_NVP(_params7);
    ar & BOOST_SERIALIZATION_NVP(_params8);
    ar & BOOST_SERIALIZATION_NVP(_params9);
    ar & BOOST_SERIALIZATION_NVP(_params10);
    ar & BOOST_SERIALIZATION_NVP(_radLength);
    ar & BOOST_SERIALIZATION_NVP(_xi);
    ar & BOOST_SERIALIZATION_NVP(_pixROCRows);
    ar & BOOST_SERIALIZATION_NVP(_pixROCCols);
    ar & BOOST_SERIALIZATION_NVP(_pixROCx);
    ar & BOOST_SERIALIZATION_NVP(_pixROCy);
    ar & BOOST_SERIALIZATION_NVP(_siliconAPVNum);
    ar & BOOST_SERIALIZATION_NVP(_level);
    ar & BOOST_SERIALIZATION_NVP(_shape);
    ar & BOOST_SERIALIZATION_NVP(_type);
    ar & BOOST_SERIALIZATION_NVP(_numnt);
    ar & BOOST_SERIALIZATION_NVP(_nt0);
    ar & BOOST_SERIALIZATION_NVP(_nt1);
    ar & BOOST_SERIALIZATION_NVP(_nt2);
    ar & BOOST_SERIALIZATION_NVP(_nt3);
    ar & BOOST_SERIALIZATION_NVP(_nt4);
    ar & BOOST_SERIALIZATION_NVP(_nt5);
    ar & BOOST_SERIALIZATION_NVP(_nt6);
    ar & BOOST_SERIALIZATION_NVP(_nt7);
    ar & BOOST_SERIALIZATION_NVP(_nt8);
    ar & BOOST_SERIALIZATION_NVP(_nt9);
    ar & BOOST_SERIALIZATION_NVP(_nt10);
    ar & BOOST_SERIALIZATION_NVP(_geographicalID);
    ar & BOOST_SERIALIZATION_NVP(_stereo);
}

template <class Archive>
void PGeometricDetExtra::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pgdes_);
}

template <class Archive>
void PGeometricDetExtra::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(_geographicalId);
    ar & BOOST_SERIALIZATION_NVP(_volume);
    ar & BOOST_SERIALIZATION_NVP(_density);
    ar & BOOST_SERIALIZATION_NVP(_weight);
    ar & BOOST_SERIALIZATION_NVP(_copy);
    ar & BOOST_SERIALIZATION_NVP(_material);
}

template <class Archive>
void RecoIdealGeometry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pDetIds);
    ar & BOOST_SERIALIZATION_NVP(pPars);
    ar & BOOST_SERIALIZATION_NVP(pParsIndex);
    ar & BOOST_SERIALIZATION_NVP(pNumShapeParms);
    ar & BOOST_SERIALIZATION_NVP(strPars);
    ar & BOOST_SERIALIZATION_NVP(sParsIndex);
    ar & BOOST_SERIALIZATION_NVP(sNumsParms);
}

namespace cond {
namespace serialization {

template <>
struct access<CSCRecoDigiParameters>
{
    static bool equal_(const CSCRecoDigiParameters & first, const CSCRecoDigiParameters & second)
    {
        return true
            and (equal(first.pUserParOffset, second.pUserParOffset))
            and (equal(first.pUserParSize, second.pUserParSize))
            and (equal(first.pChamberType, second.pChamberType))
            and (equal(first.pfupars, second.pfupars))
        ;
    }
};

template <>
struct access<PCaloGeometry>
{
    static bool equal_(const PCaloGeometry & first, const PCaloGeometry & second)
    {
        return true
            and (equal(first.m_translation, second.m_translation))
            and (equal(first.m_dimension, second.m_dimension))
            and (equal(first.m_indexes, second.m_indexes))
            and (equal(first.m_dins, second.m_dins))
        ;
    }
};

template <>
struct access<PGeometricDet>
{
    static bool equal_(const PGeometricDet & first, const PGeometricDet & second)
    {
        return true
            and (equal(first.pgeomdets_, second.pgeomdets_))
        ;
    }
};

template <>
struct access<PGeometricDet::Item>
{
    static bool equal_(const PGeometricDet::Item & first, const PGeometricDet::Item & second)
    {
        return true
            and (equal(first._name, second._name))
            and (equal(first._ns, second._ns))
            and (equal(first._x, second._x))
            and (equal(first._y, second._y))
            and (equal(first._z, second._z))
            and (equal(first._phi, second._phi))
            and (equal(first._rho, second._rho))
            and (equal(first._a11, second._a11))
            and (equal(first._a12, second._a12))
            and (equal(first._a13, second._a13))
            and (equal(first._a21, second._a21))
            and (equal(first._a22, second._a22))
            and (equal(first._a23, second._a23))
            and (equal(first._a31, second._a31))
            and (equal(first._a32, second._a32))
            and (equal(first._a33, second._a33))
            and (equal(first._params0, second._params0))
            and (equal(first._params1, second._params1))
            and (equal(first._params2, second._params2))
            and (equal(first._params3, second._params3))
            and (equal(first._params4, second._params4))
            and (equal(first._params5, second._params5))
            and (equal(first._params6, second._params6))
            and (equal(first._params7, second._params7))
            and (equal(first._params8, second._params8))
            and (equal(first._params9, second._params9))
            and (equal(first._params10, second._params10))
            and (equal(first._radLength, second._radLength))
            and (equal(first._xi, second._xi))
            and (equal(first._pixROCRows, second._pixROCRows))
            and (equal(first._pixROCCols, second._pixROCCols))
            and (equal(first._pixROCx, second._pixROCx))
            and (equal(first._pixROCy, second._pixROCy))
            and (equal(first._siliconAPVNum, second._siliconAPVNum))
            and (equal(first._level, second._level))
            and (equal(first._shape, second._shape))
            and (equal(first._type, second._type))
            and (equal(first._numnt, second._numnt))
            and (equal(first._nt0, second._nt0))
            and (equal(first._nt1, second._nt1))
            and (equal(first._nt2, second._nt2))
            and (equal(first._nt3, second._nt3))
            and (equal(first._nt4, second._nt4))
            and (equal(first._nt5, second._nt5))
            and (equal(first._nt6, second._nt6))
            and (equal(first._nt7, second._nt7))
            and (equal(first._nt8, second._nt8))
            and (equal(first._nt9, second._nt9))
            and (equal(first._nt10, second._nt10))
            and (equal(first._geographicalID, second._geographicalID))
            and (equal(first._stereo, second._stereo))
        ;
    }
};

template <>
struct access<PGeometricDetExtra>
{
    static bool equal_(const PGeometricDetExtra & first, const PGeometricDetExtra & second)
    {
        return true
            and (equal(first.pgdes_, second.pgdes_))
        ;
    }
};

template <>
struct access<PGeometricDetExtra::Item>
{
    static bool equal_(const PGeometricDetExtra::Item & first, const PGeometricDetExtra::Item & second)
    {
        return true
            and (equal(first._geographicalId, second._geographicalId))
            and (equal(first._volume, second._volume))
            and (equal(first._density, second._density))
            and (equal(first._weight, second._weight))
            and (equal(first._copy, second._copy))
            and (equal(first._material, second._material))
        ;
    }
};

template <>
struct access<RecoIdealGeometry>
{
    static bool equal_(const RecoIdealGeometry & first, const RecoIdealGeometry & second)
    {
        return true
            and (equal(first.pDetIds, second.pDetIds))
            and (equal(first.pPars, second.pPars))
            and (equal(first.pParsIndex, second.pParsIndex))
            and (equal(first.pNumShapeParms, second.pNumShapeParms))
            and (equal(first.strPars, second.strPars))
            and (equal(first.sParsIndex, second.sParsIndex))
            and (equal(first.sNumsParms, second.sNumsParms))
        ;
    }
};

}
}

#endif
