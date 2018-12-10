/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#ifndef DataFormats_ProtonReco_ProtonTrack_h
#define DataFormats_ProtonReco_ProtonTrack_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ProtonReco/interface/ProtonTrackExtra.h"
#include "DataFormats/ProtonReco/interface/ProtonTrackExtraFwd.h"

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
      enum struct Index : unsigned short { xi, th_x, vtx_x, th_y, vtx_y, num_indices };

      /// Longitudinal fractional momentum loss
      float xi() const { return xi_; }

      /// vertical scattering angle, in rad
      float thetaX() const { return px() / p(); }

      /// horizontal scattering angle, in rad
      float thetaY() const { return py() / p(); }

      // vertex position can be obtained via TrackBase::vx() and vy() functions

      static float calculateT(double beam_mom, double proton_mom, double theta);

      /// four-momentum transfer squared, in GeV^2
      float t() const;

      float xiError() const { return error( (int)Index::xi ); }

      float thetaXError() const { return error( (int)Index::th_x ); }

      float thetaYError() const { return error( (int)Index::th_y ); }

      float vertexXError() const { return error( (int)Index::vtx_x ); }

      float vertexYError() const { return error( (int)Index::vtx_y ); }

      /// LHC sector
      enum class LHCSector { invalid = -1, sector45, sector56 };
      LHCSector lhcSector() const
      {
        if (pz() < 0.) return LHCSector::sector56;
        if (pz() > 0.) return LHCSector::sector45;
        return LHCSector::invalid;
      }

      // convenience getters for the extra information
      bool validFit() const { return pt_extra_->validFit(); }

      ProtonTrackExtra::ReconstructionMethod method() const { return pt_extra_->method(); }

      const ProtonTrackExtra::CTPPSLocalTrackLiteRefVector& contributingLocalTracks() const { return pt_extra_->contributingLocalTracks(); }

      void setProtonTrackExtra( const ProtonTrackExtraRef& ref ) { pt_extra_ = ref; }
      const ProtonTrackExtraRef& protonTrackExtra() const { return pt_extra_; }

    private:
      float xi_; ///< fractional momentum loss (positive for diffractive protons)

      ProtonTrackExtraRef pt_extra_; ///< Additional information on proton track
  };
}

#endif
