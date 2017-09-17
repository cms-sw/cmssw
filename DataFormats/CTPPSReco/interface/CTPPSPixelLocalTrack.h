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

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Matrix.h"


//----------------------------------------------------------------------------------------------------

class CTPPSPixelFittedRecHit: public CTPPSPixelRecHit
{
public:
CTPPSPixelFittedRecHit(const CTPPSPixelRecHit &hit, const math::GlobalPoint &space_point_on_det, LocalPoint residual, LocalPoint pull) :
  CTPPSPixelRecHit(hit), space_point_on_det_(space_point_on_det), residual_(residual), pull_(pull), isUsedForFit_(false), isRealHit_(false) {}
    
CTPPSPixelFittedRecHit() : CTPPSPixelRecHit(), residual_(LocalPoint(0,0)), pull_(LocalPoint(0,0)), isUsedForFit_(false), isRealHit_(false) {}
    
  virtual ~CTPPSPixelFittedRecHit() {}
    
  inline const math::GlobalPoint & getGlobalCoordinates() const { return space_point_on_det_; }
  inline float getXResidual() const { return residual_.x(); }
  inline float getYResidual() const { return residual_.y(); }
        
  inline float getXPull() const { return pull_.x(); }
  inline float getYPull() const { return pull_.y(); }
        
  inline float getXPullNormalization() const { return residual_.x() / pull_.x(); }
  inline float getYPullNormalization() const { return residual_.y() / pull_.y(); }

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
  math::GlobalPoint space_point_on_det_ ;  ///< mm
  LocalPoint residual_;  ///< mm
  LocalPoint pull_    ;  ///< normalised residual
  bool isUsedForFit_;
  bool isRealHit_;
};


class CTPPSPixelLocalTrack: public CTPPSPixelFittedRecHit
{

  public:
    ///< parameter vector size
    static constexpr int dimension = 4;
    typedef math::Error<dimension>::type CovarianceMatrix;
    typedef math::Vector<dimension>::type ParameterVector;

    ///< covariance matrix size
    static constexpr int covarianceSize = dimension * dimension;

    CTPPSPixelLocalTrack() : z0_(0), chiSquared_(0), valid_(false), numberOfPointUsedForFit_(0)
    {
    }

    CTPPSPixelLocalTrack(float z0, const ParameterVector & track_params_vector,
      const CovarianceMatrix &par_covariance_matrix, float chiSquared);

    ~CTPPSPixelLocalTrack() override {}

    inline const edm::DetSetVector<CTPPSPixelFittedRecHit>& getHits() const { return track_hits_vector_; }
    inline void addHit(unsigned int detId, const CTPPSPixelFittedRecHit &hit)
    {
      track_hits_vector_.find_or_insert(detId).push_back(hit);
      if(hit.getIsUsedForFit()) ++numberOfPointUsedForFit_;
    }

    inline float getX0() const { return track_params_vector_[0]; }
    inline float getX0Sigma() const { return sqrt(covarianceMatrixElement(0, 0)); }
    inline float getX0Variance() const { return covarianceMatrixElement(0, 0); }

    inline float getY0() const { return track_params_vector_[1]; }
    inline float getY0Sigma() const { return sqrt(covarianceMatrixElement(1, 1)); }
    inline float getY0Variance() const { return covarianceMatrixElement(1, 1); }

    inline float getZ0() const { return z0_; }
    inline void setZ0(float z0) { z0_ = z0; }

    inline float getTx() const { return track_params_vector_[2]; }
    inline float getTxSigma() const { return sqrt(covarianceMatrixElement(2, 2)); }

    inline float getTy() const { return track_params_vector_[3]; }
    inline float getTySigma() const { return sqrt(covarianceMatrixElement(3, 3)); }

    inline math::GlobalVector getDirectionVector() const
    {
      math::GlobalVector vect(getTx(), getTy(), 1);
      return vect.unit();
    }

    ParameterVector getParameterVector() const;

    CovarianceMatrix getCovarianceMatrix() const;

    inline float getChiSquared() const { return chiSquared_; }

    inline float getChiSquaredOverNDF() const { 
      if(numberOfPointUsedForFit_<= 2.) return -999.;
      else return chiSquared_ / (2*numberOfPointUsedForFit_ - dimension); 
    }

    inline int getNDF() const {return (2*numberOfPointUsedForFit_ - dimension); }

    /// returns the point from which the track is passing by at the selected z
    inline math::GlobalPoint getTrackPoint(float z) const 
    {
      float delta_z = z - z0_;
      return math::GlobalPoint(
        track_params_vector_[0] + track_params_vector_[2] * delta_z,
        track_params_vector_[1] + track_params_vector_[3] * delta_z,
        z);
    }

    inline math::GlobalPoint getTrackCentrePoint()
    {
      return math::GlobalPoint(track_params_vector_[0], track_params_vector_[1], z0_);
    }

    math::Matrix<2,2>::type trackPointInterpolationCovariance(float z) const;

    inline bool isValid() const { return valid_; }

    inline void setValid(bool valid) { valid_ = valid; }

    bool operator< (const CTPPSPixelLocalTrack &r);
    
  private:
    inline const float& covarianceMatrixElement(int i, int j) const
    {
      return par_covariance_matrix_[i * dimension + j];
    }

    inline float& covarianceMatrixElement(int i, int j)
    {
      return par_covariance_matrix_[i * dimension + j];
    }

    edm::DetSetVector<CTPPSPixelFittedRecHit> track_hits_vector_;

    /// track parameters: (x0, y0, tx, ty); x = x0 + tx*(z-z0) ...
    float track_params_vector_[dimension];

    /// z where x0 and y0 are evaluated.
    /// filled from TotemRPGeometry::GetRPGlobalTranslation
    float z0_; 

    float par_covariance_matrix_[covarianceSize];
  
    /// fit chi^2
    float chiSquared_;

    /// fit valid?
    bool valid_;

    int numberOfPointUsedForFit_;

};

#endif

