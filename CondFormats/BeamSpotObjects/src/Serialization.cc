
#include "CondFormats/BeamSpotObjects/src/headers.h"

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/export.hpp>

#include "CondFormats/Serialization/interface/Equal.h"
#include "CondFormats/Serialization/interface/Instantiate.h"

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
COND_SERIALIZATION_INSTANTIATE(BeamSpotObjects);

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
COND_SERIALIZATION_INSTANTIATE(SimBeamSpotObjects);

#include "CondFormats/BeamSpotObjects/src/SerializationManual.h"
