/****************************************************************************
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 ****************************************************************************/

#ifndef DataFormats_ProtonReco_ProtonTrack_h
#define DataFormats_ProtonReco_ProtonTrack_h

#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

namespace reco
{
  class ProtonTrack
  {
    public:
      /// parameter dimension
      enum { dimension = 5 };
      /// indices to the covariance matrix
      enum struct Index : unsigned short { xi, th_x, vtx_x, th_y, vtx_y, num_indices = dimension };
      /// dimension-parameter covariance matrix
      typedef math::Error<dimension>::type CovarianceMatrix;
      /// spatial vector
      typedef math::XYZVector Vector;
      /// point in the space
      typedef math::XYZPoint Point;

      using CTPPSLocalTrackLiteRefVector = std::vector<edm::Ref<std::vector<CTPPSLocalTrackLite> > >;

      /// Type of reconstruction for this track
      enum class ReconstructionMethod { invalid = -1, singleRP, multiRP };

    public:
      /// Default constructor
      ProtonTrack();
      /// Constructor from refit parameters, fitted vertex and momentum, and longitudinal fractional momentum loss
      ProtonTrack( double chi2, double ndof, const Point& vtx, const Vector& momentum, float xi, const CovarianceMatrix& cov = CovarianceMatrix() );

      const Point& vertex() const { return vertex_; }
      float vx() const { return vertex_.x(); }
      float vy() const { return vertex_.y(); }
      float vz() const { return vertex_.z(); }
      const Vector& momentum() const { return momentum_; }
      float p() const { return momentum_.R(); }
      float px() const { return momentum_.x(); }
      float py() const { return momentum_.y(); }
      float pz() const { return momentum_.z(); }

      /// chi-squared of the fit
      float chi2() const { return chi2_; }
      /// number of degrees of freedom for the track fit
      unsigned int ndof() const { return ndof_; }
      /// chi-squared divided by ndof (or chi-squared * 1e6 if ndof is zero)
      float normalizedChi2() const {
        return ( ndof_ != 0 ) ? chi2_ / ndof_ : chi2_ * 1.e6;
      }

      /// longitudinal fractional momentum loss
      float xi() const { return xi_; }
      /// vertical scattering angle, in rad
      float thetaX() const { return px() / p(); }
      /// horizontal scattering angle, in rad
      float thetaY() const { return py() / p(); }

      // vertex position can be obtained via TrackBase::vx() and vy() functions

      /// uncertainty on longitudinal fractional momentum loss
      float xiError() const { return error( Index::xi ); }
      /// uncertainty on fitted momentum horizontal angle opening
      float thetaXError() const { return error( Index::th_x ); }
      /// uncertainty on fitted momentum vertical angle opening
      float thetaYError() const { return error( Index::th_y ); }
      /// uncertainty on fitted vertex horizontal position
      float vertexXError() const { return error( Index::vtx_x ); }
      /// uncertainty on fitted vertex vertical position
      float vertexYError() const { return error( Index::vtx_y ); }

      /// proton mass in GeV
      static float mass() { return mass_; }

      /// compute the squared four-momentum transfer from incident and scattered momenta, and angular information
      static float calculateT( double beam_mom, double proton_mom, double theta );

      /// four-momentum transfer squared, in GeV^2
      float t() const;

      /// conveniece getters for time of proton arrival at RPs
      float time() const { return t_; }
      float timeError() const { return t_err_; }

      /// Set the flag for the fit validity
      void setValidFit( bool valid = true ) { valid_fit_ = valid; }
      /// Flag for the fit validity
      bool validFit() const { return valid_fit_; }

      /// Set the reconstruction method for this track
      void setMethod( const ReconstructionMethod& method ) { method_ = method; }
      /// Reconstruction method for this track
      ReconstructionMethod method() const { return method_; }

      /// Store the list of RP tracks that contributed to this global track
      void setContributingLocalTracks( const CTPPSLocalTrackLiteRefVector &v ) { contributing_local_tracks_ = v; }
      /// List of RP tracks that contributed to this global track
      const CTPPSLocalTrackLiteRefVector& contributingLocalTracks() const { return contributing_local_tracks_; }

      /// LHC sector
      enum class LHCSector { invalid = -1, sector45, sector56 };
      LHCSector lhcSector() const
      {
        if ( pz() < 0. ) return LHCSector::sector56;
        if ( pz() > 0. ) return LHCSector::sector45;
        return LHCSector::invalid;
      }

    private:
      static constexpr float mass_ = 0.938272046; ///< proton mass, GeV
      static constexpr float massSquared_ = 0.88035443; ///< proton mass squared, GeV^2

      /*inline size_t covIndex( Index i, Index j ) const {
        const size_t a = ( i <= j ? i : j ), b = ( i <= j ? j : i );
        return b * 0.5*( b + 1 ) + a;
      }*/
      double error( Index i ) const {
        return sqrt( covariance_( (unsigned int)i, (unsigned int)i ) );
      }

      /// reconstructed vertex position at z/s = 0
      Point vertex_;
      /// reconstructed momentum vector
      Vector momentum_;
      /// reconstructed time at forward detectors
      float t_;
      /// uncertainty on reconstructed time at forward detectors
      float t_err_;
      /// fractional momentum loss (positive for diffractive protons)
      float xi_;
      /// 5x5 covariance matrix
      CovarianceMatrix covariance_;
      /// chi-squared
      float chi2_;
      /// number of degrees of freedom
      unsigned int ndof_;
      bool valid_fit_;
      ReconstructionMethod method_;
      CTPPSLocalTrackLiteRefVector contributing_local_tracks_;
  };
}

#endif
