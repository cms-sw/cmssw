/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

#include <set>

using namespace reco;

ProtonTrack::ProtonTrack() :
  xi_( 0. )
{}

ProtonTrack::ProtonTrack( double chi2, double ndof, const Point& vtx, const Vector& momentum, float xi, const CovarianceMatrix& cov ) :
  Track( chi2, ndof, vtx, momentum, +1, cov ), xi_( xi )
{}

float ProtonTrack::calculateT( double beam_mom, double proton_mom, double theta )
{
  const double m = 0.938; // GeV
  //FIXME necessarily hardcoded? may be moved to a static const(expr) value? or method argument?

  const double t0 = 2.*m*m + 2.*beam_mom*proton_mom - 2.*sqrt( (m*m + beam_mom*beam_mom) * (m*m + proton_mom*proton_mom) );
  const double S = sin(theta/2.);
  return t0 - 4. * beam_mom * proton_mom * S*S;
}

float ProtonTrack::t() const
{
  const double beam_mom = p() / (1.-xi());
  const double theta = sqrt(thetaX()*thetaX() + thetaY()*thetaY());
  return calculateT(beam_mom, p(), theta);
}
