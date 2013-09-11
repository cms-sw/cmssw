#ifndef CondFormats_BeamSpotObjects_Serialization_H
#define CondFormats_BeamSpotObjects_Serialization_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>

#include "CondFormats/External/interface/Serialization.h"

#include "../src/headers.h"

template <class Archive>
void BeamSpotObjects::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(position_);
    ar & BOOST_SERIALIZATION_NVP(sigmaZ_);
    ar & BOOST_SERIALIZATION_NVP(beamwidthX_);
    ar & BOOST_SERIALIZATION_NVP(beamwidthY_);
    ar & BOOST_SERIALIZATION_NVP(beamwidthXError_);
    ar & BOOST_SERIALIZATION_NVP(beamwidthYError_);
    ar & BOOST_SERIALIZATION_NVP(dxdz_);
    ar & BOOST_SERIALIZATION_NVP(dydz_);
    ar & BOOST_SERIALIZATION_NVP(covariance_);
    ar & BOOST_SERIALIZATION_NVP(type_);
    ar & BOOST_SERIALIZATION_NVP(emittanceX_);
    ar & BOOST_SERIALIZATION_NVP(emittanceY_);
    ar & BOOST_SERIALIZATION_NVP(betaStar_);
}

template <class Archive>
void SimBeamSpotObjects::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(fX0);
    ar & BOOST_SERIALIZATION_NVP(fY0);
    ar & BOOST_SERIALIZATION_NVP(fZ0);
    ar & BOOST_SERIALIZATION_NVP(fSigmaZ);
    ar & BOOST_SERIALIZATION_NVP(fbetastar);
    ar & BOOST_SERIALIZATION_NVP(femittance);
    ar & BOOST_SERIALIZATION_NVP(fPhi);
    ar & BOOST_SERIALIZATION_NVP(fAlpha);
    ar & BOOST_SERIALIZATION_NVP(fTimeOffset);
}

namespace cond {
namespace serialization {

template <>
struct access<BeamSpotObjects>
{
    static bool equal_(const BeamSpotObjects & first, const BeamSpotObjects & second)
    {
        return true
            and (equal(first.position_, second.position_))
            and (equal(first.sigmaZ_, second.sigmaZ_))
            and (equal(first.beamwidthX_, second.beamwidthX_))
            and (equal(first.beamwidthY_, second.beamwidthY_))
            and (equal(first.beamwidthXError_, second.beamwidthXError_))
            and (equal(first.beamwidthYError_, second.beamwidthYError_))
            and (equal(first.dxdz_, second.dxdz_))
            and (equal(first.dydz_, second.dydz_))
            and (equal(first.covariance_, second.covariance_))
            and (equal(first.type_, second.type_))
            and (equal(first.emittanceX_, second.emittanceX_))
            and (equal(first.emittanceY_, second.emittanceY_))
            and (equal(first.betaStar_, second.betaStar_))
        ;
    }
};

template <>
struct access<SimBeamSpotObjects>
{
    static bool equal_(const SimBeamSpotObjects & first, const SimBeamSpotObjects & second)
    {
        return true
            and (equal(first.fX0, second.fX0))
            and (equal(first.fY0, second.fY0))
            and (equal(first.fZ0, second.fZ0))
            and (equal(first.fSigmaZ, second.fSigmaZ))
            and (equal(first.fbetastar, second.fbetastar))
            and (equal(first.femittance, second.femittance))
            and (equal(first.fPhi, second.fPhi))
            and (equal(first.fAlpha, second.fAlpha))
            and (equal(first.fTimeOffset, second.fTimeOffset))
        ;
    }
};

}
}

#endif
