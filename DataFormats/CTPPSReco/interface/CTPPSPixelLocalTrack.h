/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/

#ifndef DataFormats_CTPPSReco_CTPPSPixelLocalTrack_H
#define DataFormats_CTPPSReco_CTPPSPixelLocalTrack_H

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelRecHit.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSPixelDetId.h"


#include "TVector3.h"
#include "TMatrixD.h"
#include "TVectorD.h"

//----------------------------------------------------------------------------------------------------

class CTPPSPixelLocalTrack
{
  public:
    class CTPPSPixelFittedRecHit: public CTPPSPixelRecHit
    {
      public:
        CTPPSPixelFittedRecHit(const CTPPSPixelRecHit &hit, const TVector3 &space_point_on_det, std::pair<double,double> residual, std::pair<double,double> pull) :
            CTPPSPixelRecHit(hit), space_point_on_det_(space_point_on_det), residual_(residual), pull_(pull), isUsedForFit_(false), isRealHit_(false) {}
    
        CTPPSPixelFittedRecHit() : CTPPSPixelRecHit(), residual_(std::pair<double,double>(0,0)), pull_(std::pair<double,double>(0,0)), isUsedForFit_(false), isRealHit_(false) {}
    
        virtual ~CTPPSPixelFittedRecHit() {}
    
        inline const TVector3 & getGlobalCoordinates() const { return space_point_on_det_; }
        inline void setGlobalCoordinates(const TVector3 & space_point_on_det) { space_point_on_det_ = space_point_on_det; }
    
        inline double getXResidual() const { return residual_.first; }
        inline double getYResidual() const { return residual_.second; }
        inline void setXResidual(double residual) { residual_.first = residual; }
        inline void setYResidual(double residual) { residual_.second = residual; }
    
        inline double getXPull() const { return pull_.first; }
        inline double getYPull() const { return pull_.second; }
        inline void setXPull(double pull) { pull_.first = pull; }
        inline void setYPull(double pull) { pull_.second = pull; }
    
        inline double getXPullNormalization() const { return residual_.first / pull_.first; }
        inline double getYPullNormalization() const { return residual_.second / pull_.second; }

        inline void setIsUsedForFit(bool usedForFit) {
            if(usedForFit) isRealHit_ = true; 
            isUsedForFit_ = usedForFit; 
        }
        inline bool getIsUsedForFit() const { return isUsedForFit_; }

        inline void setIsRealHit(bool realHit) { 
            if(!realHit) isUsedForFit_ = false;
            isRealHit_ = realHit; 
        }
        inline bool getIsRealHit() const { return isRealHit_; }
    
      private:
        TVector3 space_point_on_det_ ;  ///< mm
        std::pair<double,double> residual_;  ///< mm
        std::pair<double,double> pull_    ;  ///< normalised residual
        bool isUsedForFit_;
        bool isRealHit_;
    };

  public:
    ///< parameter vector size
    static const int dimension = 4;

    ///< covariance matrix size
    static const int covarianceSize = dimension * dimension;

    CTPPSPixelLocalTrack() : z0_(0), chiSquared_(0), valid_(false), numberOfPointUsedForFit_(0)
    {
    }

    CTPPSPixelLocalTrack(double z0, const TVectorD &track_params_vector,
      const TMatrixD &par_covariance_matrix, double chiSquared);

    virtual ~CTPPSPixelLocalTrack() {}

    inline const edm::DetSetVector<CTPPSPixelFittedRecHit>& getHits() const { return track_hits_vector_; }
    inline void addHit(unsigned int detId, const CTPPSPixelFittedRecHit &hit)
    {
      track_hits_vector_.find_or_insert(detId).push_back(hit);
      if(hit.getIsUsedForFit()) ++numberOfPointUsedForFit_;
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

    inline double getChiSquaredOverNDF() const { return chiSquared_ / (2*numberOfPointUsedForFit_ - 4); }

    inline int getNDF() const {return (2*numberOfPointUsedForFit_ - 4); }

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

    friend bool operator< (const CTPPSPixelLocalTrack &l, const CTPPSPixelLocalTrack &r);

  private:
    inline const double& CovarianceMatrixElement(int i, int j) const
    {
      return par_covariance_matrix_[i * dimension + j];
    }

    inline double& CovarianceMatrixElement(int i, int j)
    {
      return par_covariance_matrix_[i * dimension + j];
    }

    edm::DetSetVector<CTPPSPixelFittedRecHit> track_hits_vector_;

    /// track parameters: (x0, y0, tx, ty); x = x0 + tx*(z-z0) ...
    double track_params_vector_[dimension];

    /// z where x0 and y0 are evaluated.
    /// filled from TotemRPGeometry::GetRPGlobalTranslation
    double z0_; 

    double par_covariance_matrix_[covarianceSize];
  
    /// fit chi^2
    double chiSquared_;

    /// fit valid?
    bool valid_;

    int numberOfPointUsedForFit_;

};

#endif

