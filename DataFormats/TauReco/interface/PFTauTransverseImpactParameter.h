#ifndef DataFormats_TauReco_PFTauTransverseImpactParameter_h
#define DataFormats_TauReco_PFTauTransverseImpactParameter_h

/* class PFTauTransverseImpactParameter
 * 
 * Stores information on the impact paramters and flight length of the hadronic decay of a tau lepton
 *
 * author: Ian M. Nugent
 * This work is based on the impact parameter work by Rosamaria Venditti and reconstructing the 3 prong taus.
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * work was continued by Ian M. Nugent and Vladimir Cherepanov.
 * Thanks goes to Christian Veelken and Evan Klose Friis for their help and suggestions. 
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TVector3.h"

namespace reco 
{
  class PFTauTransverseImpactParameter 
  {
    enum { dimension = 3 };
    enum { covarianceSize = dimension * ( dimension + 1 ) / 2 };

   public:
    typedef math::Error<dimension>::type CovMatrix;
    typedef math::XYZPoint Point;
    typedef math::XYZVector Vector;

    PFTauTransverseImpactParameter(){}
    /// constructor from values
    PFTauTransverseImpactParameter(const Point&, double, double, const Point&, double, double, const VertexRef&);
    PFTauTransverseImpactParameter(const Point&, double, double, const Point&, double, double, const VertexRef&, const Point&, double, const VertexRef&);
    
    virtual ~PFTauTransverseImpactParameter(){}
    PFTauTransverseImpactParameter* clone() const;
    
    const Point&     dxy_PCA() const { return pca_; }
    double           dxy() const { return dxy_; }
    double           dxy_error() const { return dxy_error_; }
    double           dxy_Sig() const { return ( dxy_error_ != 0 ) ? (dxy_/dxy_error_) : 0.; }
    const Point&     ip3d_PCA() const { return pca3d_; }
    double           ip3d() const { return ip3d_; }
    double           ip3d_error() const { return ip3d_error_; }
    double           ip3d_Sig() const { return ( ip3d_error_ != 0 ) ? (ip3d_/ip3d_error_) : 0.; }
    const VertexRef& primaryVertex() const { return PV_; }
    Point            primaryVertexPos() const; 
    CovMatrix        primaryVertexCov() const;
    bool             hasSecondaryVertex() const { return hasSV_; }
    const Vector&    flightLength() const;
    double           flightLengthSig() const;
    CovMatrix        flightLengthCov() const;
    const VertexRef& secondaryVertex() const { return SV_; }
    Point            secondaryVertexPos() const;
    CovMatrix        secondaryVertexCov() const;
    
  private:
    Point      pca_;
    double     dxy_;
    double     dxy_error_;
    Point      pca3d_;
    double     ip3d_;
    double     ip3d_error_;
    VertexRef  PV_;
    bool       hasSV_;
    Vector     FlightLength_;
    double     FlightLengthSig_;
    VertexRef  SV_;    
  };
}

#endif
