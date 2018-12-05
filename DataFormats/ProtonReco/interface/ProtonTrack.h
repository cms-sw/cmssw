/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#ifndef DataFormats_ProtonReco_ProtonTrack_h
#define DataFormats_ProtonReco_ProtonTrack_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ProtonReco/interface/ProtonTrackExtraFwd.h"
#include <set>

namespace reco
{
  class ProtonTrack : public Track
  {
    public:
      /// Default constructor
      ProtonTrack();
      /// Constructor from refit parameters, fitted vertex and direction, and longitudinal fractional momentum loss
      ProtonTrack( double chi2, double ndof, const Point& vtx, const Vector& dir, float xi, const CovarianceMatrix& cov = CovarianceMatrix() );

      /// Indices to the covariance matrix
      enum struct Index : unsigned short { xi, th_x, th_y, vtx_y, num_indices };

      /// Longitudinal fractional momentum loss
      float xi() const { return xi_; }
      /// Absolute uncertainty on longitudinal fractional momentum loss
      float xiError() const { return error( (int)Index::xi ); }

      void setProtonTrackExtra( const ProtonTrackExtraRef& ref ) { pt_extra_ = ref; }
      const ProtonTrackExtraRef& protonTrackExtra() const { return pt_extra_; }

    private:
      float xi_; ///< Longitudinal fractional momentum loss
      ProtonTrackExtraRef pt_extra_; ///< Additional information on proton track
  };
}

#endif
