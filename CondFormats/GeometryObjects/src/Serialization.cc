
#include "CondFormats/GeometryObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

template <class Archive>
void CSCRecoDigiParameters::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pUserParOffset);
    ar & BOOST_SERIALIZATION_NVP(pUserParSize);
    ar & BOOST_SERIALIZATION_NVP(pChamberType);
    ar & BOOST_SERIALIZATION_NVP(pfupars);
}
COND_SERIALIZATION_INSTANTIATE(CSCRecoDigiParameters);

template <class Archive>
void PCaloGeometry::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(m_translation);
    ar & BOOST_SERIALIZATION_NVP(m_dimension);
    ar & BOOST_SERIALIZATION_NVP(m_indexes);
    ar & BOOST_SERIALIZATION_NVP(m_dins);
}
COND_SERIALIZATION_INSTANTIATE(PCaloGeometry);

template <class Archive>
void PGeometricDet::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pgeomdets_);
}
COND_SERIALIZATION_INSTANTIATE(PGeometricDet);

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
COND_SERIALIZATION_INSTANTIATE(PGeometricDet::Item);

template <class Archive>
void PGeometricDetExtra::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(pgdes_);
}
COND_SERIALIZATION_INSTANTIATE(PGeometricDetExtra);

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
COND_SERIALIZATION_INSTANTIATE(PGeometricDetExtra::Item);

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
COND_SERIALIZATION_INSTANTIATE(RecoIdealGeometry);

#include "CondFormats/GeometryObjects/src/SerializationManual.h"
