/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

#include <set>

using namespace reco;

ProtonTrack::ProtonTrack() :
  t_( 0. ), t_err_( 0. ), xi_( 0. ), chi2_( 0. ), ndof_( 0 )
{}

ProtonTrack::ProtonTrack( double chi2, double ndof, const Point& vtx, const Vector& momentum, float xi, const CovarianceMatrix& cov ) :
  vertex_( vtx ), momentum_( momentum ),
  t_( 0. ), t_err_( 0. ), xi_( xi ),
  covariance_( cov ), chi2_( chi2 ), ndof_( ndof )
{}

float ProtonTrack::calculateT( double beam_mom, double proton_mom, double theta )
{
  const double t0 = 2.*( massSquared_+beam_mom*proton_mom-sqrt( ( massSquared_+beam_mom*beam_mom ) * ( massSquared_+proton_mom*proton_mom ) ) );
  const double S = sin(theta/2.);
  return t0 - 4. * beam_mom * proton_mom * S*S;
}

float ProtonTrack::t() const
{
  const double beam_mom = p() / (1.-xi());
  const double theta = std::hypot( thetaX(), thetaY() );
  return calculateT( beam_mom, p(), theta );
}
