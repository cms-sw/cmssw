/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include "DataFormats/ProtonReco/interface/ForwardProton.h"

#include <set>

using namespace reco;

ForwardProton::ForwardProton() :
  time_( 0. ), time_err_( 0. ), xi_( 0. ), chi2_( 0. ), ndof_( 0 ),
  valid_fit_( false ), method_( ReconstructionMethod::invalid )
{}

ForwardProton::ForwardProton( double chi2, double ndof, const Point& vtx, const Vector& momentum, float xi,
                              const CovarianceMatrix& cov, ReconstructionMethod method,
                              const CTPPSLocalTrackLiteRefVector& local_tracks, bool valid ) :
  vertex_( vtx ), momentum_( momentum ),
  time_( 0. ), time_err_( 0. ), xi_( xi ),
  covariance_( cov ), chi2_( chi2 ), ndof_( ndof ),
  valid_fit_( valid ), method_( method ), contributing_local_tracks_( local_tracks )
{}

float
ForwardProton::calculateT( double beam_mom, double proton_mom, double theta )
{
  const double t0 = 2.*( massSquared_+beam_mom*proton_mom-sqrt( ( massSquared_+beam_mom*beam_mom ) * ( massSquared_+proton_mom*proton_mom ) ) );
  const double S = sin(theta/2.);
  return t0 - 4. * beam_mom * proton_mom * S*S;
}

float
ForwardProton::t() const
{
  const double beam_mom = p() / (1.-xi());
  const double theta = std::hypot( thetaX(), thetaY() );
  return calculateT( beam_mom, p(), theta );
}
