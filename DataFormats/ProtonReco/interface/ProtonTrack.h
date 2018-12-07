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

      /// Constructor from refit parameters, fitted vertex and momentum, and longitudinal fractional momentum loss
      ProtonTrack( double chi2, double ndof, const Point& vtx, const Vector& momentum, float xi, const CovarianceMatrix& cov = CovarianceMatrix() );

      /// Indices to the covariance matrix
      // TODO: what is num_indeces, where used ??
      // TODO: add (formally) also vtx_x ?? the base class defines dimension = 5
      enum struct Index : unsigned short { xi, th_x, th_y, vtx_y, num_indices };

      /// Longitudinal fractional momentum loss
      float xi() const { return xi_; }

      /// Absolute uncertainty on longitudinal fractional momentum loss
      // TODO: rename error -> uncertainty ?
      float xiError() const { return error( (int)Index::xi ); }

      void setProtonTrackExtra( const ProtonTrackExtraRef& ref ) { pt_extra_ = ref; }
      const ProtonTrackExtraRef& protonTrackExtra() const { return pt_extra_; }

      // TODO: add getters for theta*_x and theta*_y, ...
      // TODO: add getter for t

      // TODO: add convenience getters for the extra data (as done in reco::Track) ??

    private:
      float xi_; ///< Longitudinal fractional momentum loss

      ProtonTrackExtraRef pt_extra_; ///< Additional information on proton track
  };
}

#endif
