/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

#include <set>

using namespace reco;

ProtonTrack::ProtonTrack() :
  xi_( 0. )
{}

ProtonTrack::ProtonTrack( double chi2, double ndof, const Point& vtx, const Vector& dir, float xi, const CovarianceMatrix& cov ) :
  Track( chi2, ndof, vtx, dir, +1, cov ), xi_( xi )
{}

