/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *   Hubert Niewiadomski
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSReco_TotemRPLocalTrack
#define DataFormats_CTPPSReco_TotemRPLocalTrack

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPRecHit.h"

#include "TVector3.h"
#include "TMatrixD.h"
#include "TVectorD.h"

//----------------------------------------------------------------------------------------------------

/**
 *\brief A track fit through a single RP.
 *
 * x = x0+tx*(z-z0) y = ...
 *
 * z0 is defined below
 * x any y refer to the global (x, y) system with the beam at (x = 0, y = 0).
 * Only VALID tracks (IsValid()==true) can be later used for physics reconstruction!
 **/
class TotemRPLocalTrack
{
  public:
    class FittedRecHit: public TotemRPRecHit
    {
      public:
        FittedRecHit(const TotemRPRecHit &hit, const TVector3 &space_point_on_det, double residual, double pull) :
            TotemRPRecHit(hit), space_point_on_det_(space_point_on_det), residual_(residual), pull_(pull) {}
    
        FittedRecHit() : TotemRPRecHit(), residual_(0), pull_(0) {}
    
        virtual ~FittedRecHit() {}
    
        inline const TVector3 & getGlobalCoordinates() const { return space_point_on_det_; }
        inline void setGlobalCoordinates(const TVector3 & space_point_on_det) { space_point_on_det_ = space_point_on_det; }
    
        inline double getResidual() const { return residual_; }
        inline void setResidual(double residual) { residual_ = residual; }
    
        inline double getPull() const { return pull_; }
        inline void setPull(double pull) { pull_ = pull; }
    
        inline double getPullNormalization() const { return residual_ / pull_; }
    
      private:
        TVector3 space_point_on_det_; ///< mm
        double residual_;             ///< mm
        double pull_;                 ///< normalised residual
    };

  public:
    ///< parameter vector size
    static const int dimension = 4;

    ///< covariance matrix size
    static const int covarianceSize = dimension * dimension;

    TotemRPLocalTrack() : z0_(0), chiSquared_(0), valid_(false)
    {
    }

    TotemRPLocalTrack(double z0, const TVectorD &track_params_vector,
      const TMatrixD &par_covariance_matrix, double chiSquared);

    virtual ~TotemRPLocalTrack() {}

    inline const edm::DetSetVector<FittedRecHit>& getHits() const { return track_hits_vector_; }
    inline void addHit(unsigned int detId, const FittedRecHit &hit)
    {
      track_hits_vector_.find_or_insert(detId).push_back(hit);
    }

    inline double getX0() const { return track_params_vector_[0]; }
    inline double getX0Sigma() const { return sqrt(CovarianceMatrixElement(0, 0)); }
    inline double getX0Variance() const { return CovarianceMatrixElement(0, 0); }

    inline double getY0() const { return track_params_vector_[1]; }
    inline double getY0Sigma() const { return sqrt(CovarianceMatrixElement(1, 1)); }
    inline double getY0Variance() const { return CovarianceMatrixElement(1, 1); }

    inline double getZ0() const { return z0_; }
    inline void setZ0(double z0) { z0_ = z0; }

    inline double getTx() const { return track_params_vector_[2]; }
    inline double getTxSigma() const { return sqrt(CovarianceMatrixElement(2, 2)); }

    inline double getTy() const { return track_params_vector_[3]; }
    inline double getTySigma() const { return sqrt(CovarianceMatrixElement(3, 3)); }

    inline TVector3 getDirectionVector() const
    {
      TVector3 vect(getTx(), getTy(), 1);
      vect.SetMag(1.0);
      return vect;
    }

    TVectorD getParameterVector() const;
    void setParameterVector(const TVectorD & track_params_vector);

    TMatrixD getCovarianceMatrix() const;
    void setCovarianceMatrix(const TMatrixD &par_covariance_matrix);

    inline double getChiSquared() const { return chiSquared_; }
    inline void setChiSquared(double & chiSquared) { chiSquared_ = chiSquared; }

    inline double getChiSquaredOverNDF() const { return chiSquared_ / (track_hits_vector_.size() - 4); }

    /// returns (x, y) vector
    inline TVector2 getTrackPoint(double z) const 
    {
      double delta_z = z - z0_;
      return TVector2(
        track_params_vector_[0] + track_params_vector_[2] * delta_z,
        track_params_vector_[1] + track_params_vector_[3] * delta_z);
    }

    inline TVector3 getTrackCentrePoint()
    {
      return TVector3(track_params_vector_[0], track_params_vector_[1], z0_);
    }

    TMatrixD trackPointInterpolationCovariance(double z) const;

    inline bool isValid() const { return valid_; }

    inline void setValid(bool valid) { valid_ = valid; }

    friend bool operator< (const TotemRPLocalTrack &l, const TotemRPLocalTrack &r);

  private:
    inline const double& CovarianceMatrixElement(int i, int j) const
    {
      return par_covariance_matrix_[i * dimension + j];
    }

    inline double& CovarianceMatrixElement(int i, int j)
    {
      return par_covariance_matrix_[i * dimension + j];
    }

    edm::DetSetVector<FittedRecHit> track_hits_vector_;

    /// track parameters: (x0, y0, tx, ty); x = x0 + tx*(z-z0) ...
    double track_params_vector_[dimension];

    /// z where x0 and y0 are evaluated.
    /// filled from CTPPSGeometry::getRPTranslation
    double z0_; 

    double par_covariance_matrix_[covarianceSize];
  
    /// fit chi^2
    double chiSquared_;

    /// fit valid?
    bool valid_;
};

#endif
