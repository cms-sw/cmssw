#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TVectorT.h"
using namespace reco;

PFTauTransverseImpactParameter::PFTauTransverseImpactParameter(const Point& pca, double thedxy, double thedxy_error, const VertexRef& PV)
  : pca_(pca),
    dxy_(thedxy),
    dxy_error_(thedxy_error),
    PV_(PV),
    hasSV_(false)
{}

PFTauTransverseImpactParameter::PFTauTransverseImpactParameter(const Point& pca, double thedxy, double thedxy_error, const VertexRef& PV,
							       const Point& theFlightLength, const VertexRef& SV)
  : pca_(pca),
    dxy_(thedxy),
    dxy_error_(thedxy_error),
    PV_(PV),
    hasSV_(true),
    FlightLength_(theFlightLength),
    SV_(SV)
{}

PFTauTransverseImpactParameter* PFTauTransverseImpactParameter::clone() const
{
  return new PFTauTransverseImpactParameter(*this);
}

PFTauTransverseImpactParameter::Point PFTauTransverseImpactParameter::primaryVertexPos() const 
{ 
  if ( PV_.isNonnull() ) return PV_->position(); 
  else return PFTauTransverseImpactParameter::Point(0.,0.,0.);
}

PFTauTransverseImpactParameter::CovMatrix PFTauTransverseImpactParameter::primaryVertexCov() const
{
  CovMatrix cov;
  for ( int i = 0; i < dimension; ++i ) {
    for ( int j = 0; j < dimension; ++j ) {
      cov(i,j) = PV_->covariance(i,j);
    }
  }
  return cov;
}

const PFTauTransverseImpactParameter::Vector& PFTauTransverseImpactParameter::flightLength() const 
{
  return FlightLength_;
}

double PFTauTransverseImpactParameter::flightLengthSig() const{
  if ( hasSV_ ) {
    VertexDistance3D vtxdist;
    return vtxdist.distance(*PV_, *SV_).significance(); // transforms using the jacobian then computes distance/uncertainty 
  }
  return -1.;
}

PFTauTransverseImpactParameter::Point PFTauTransverseImpactParameter::secondaryVertexPos() const{ 
  if ( hasSV_ ) return SV_->position(); 
  else return PFTauTransverseImpactParameter::Point(0.,0.,0.);
}

PFTauTransverseImpactParameter::CovMatrix PFTauTransverseImpactParameter::secondaryVertexCov() const{
  CovMatrix cov;
  if ( !hasSV_ ) return cov;
  for ( int i = 0; i < dimension; ++i ) {
    for ( int j = 0; j < dimension; ++j ) {
      cov(i,j) = SV_->covariance(i,j);
    }
  }
  return cov;
}

PFTauTransverseImpactParameter::CovMatrix PFTauTransverseImpactParameter::flightLengthCov() const{
  CovMatrix cov;
  const CovMatrix& sv = secondaryVertexCov();
  const CovMatrix& pv = primaryVertexCov();
  for ( int i = 0; i < dimension; ++i ) {
    for ( int j = 0; j < dimension; ++j ) {
      cov(i,j) = sv(i,j) + pv(i,j);
    }
  }
  return cov;
}
