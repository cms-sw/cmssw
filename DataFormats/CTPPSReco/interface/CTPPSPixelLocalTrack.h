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

#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/Math/interface/Matrix.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
//----------------------------------------------------------------------------------------------------

class CTPPSPixelFittedRecHit: public CTPPSPixelRecHit
{
 public:
 CTPPSPixelFittedRecHit(const CTPPSPixelRecHit &hit, const GlobalPoint &space_point_on_det, const LocalPoint& residual, const LocalPoint& pull) :
  CTPPSPixelRecHit(hit), space_point_on_det_(space_point_on_det), residual_(residual), pull_(pull), isUsedForFit_(false), isRealHit_(false) {}
  
 CTPPSPixelFittedRecHit() : CTPPSPixelRecHit(), residual_(LocalPoint(0,0)), pull_(LocalPoint(0,0)), isUsedForFit_(false), isRealHit_(false) {}
    
  virtual ~CTPPSPixelFittedRecHit() {}
    
  inline const GlobalPoint & getGlobalCoordinates() const { return space_point_on_det_; }
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
  GlobalPoint space_point_on_det_ ;  ///< mm
  LocalPoint residual_;  ///< mm
  LocalPoint pull_    ;  ///< normalised residual
  bool isUsedForFit_;
  bool isRealHit_;
};


class CTPPSPixelLocalTrack
{

  public:
  
  enum TrackPar {x0 = 0, y0 = 1, tx = 2, ty = 3}; 

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

    ~CTPPSPixelLocalTrack() {}

    inline const edm::DetSetVector<CTPPSPixelFittedRecHit>& getHits() const { return track_hits_vector_; }
    inline void addHit(unsigned int detId, const CTPPSPixelFittedRecHit &hit)
    {
      track_hits_vector_.find_or_insert(detId).push_back(hit);
      if(hit.getIsUsedForFit()) ++numberOfPointUsedForFit_;
    }

    inline float getX0() const { return track_params_vector_[TrackPar::x0]; }
    inline float getX0Sigma() const { return sqrt(par_covariance_matrix_[TrackPar::x0][TrackPar::x0]); }
    inline float getX0Variance() const { return par_covariance_matrix_[TrackPar::x0][TrackPar::x0]; }

    inline float getY0() const { return track_params_vector_[TrackPar::y0]; }
    inline float getY0Sigma() const { return sqrt(par_covariance_matrix_[TrackPar::y0][TrackPar::y0]); }
    inline float getY0Variance() const { return par_covariance_matrix_[TrackPar::y0][TrackPar::y0]; }

    inline float getZ0() const { return z0_; }
    inline void setZ0(float z0) { z0_ = z0; }

    inline float getTx() const { return track_params_vector_[TrackPar::tx]; }
    inline float getTxSigma() const { return sqrt(par_covariance_matrix_[TrackPar::tx][TrackPar::tx]); }

    inline float getTy() const { return track_params_vector_[TrackPar::ty]; }
    inline float getTySigma() const { return sqrt(par_covariance_matrix_[TrackPar::ty][TrackPar::ty]); }

    inline GlobalVector getDirectionVector() const
    {
      GlobalVector vect(getTx(), getTy(), 1);
      return vect.unit();
    }

    inline const ParameterVector& getParameterVector() const{
      return track_params_vector_;
    }

    inline const CovarianceMatrix& getCovarianceMatrix() const{
      return par_covariance_matrix_;
    }

    inline float getChiSquared() const { return chiSquared_; }

    inline float getChiSquaredOverNDF() const { 
      if(numberOfPointUsedForFit_<= dimension/2) return -999.;
      else return chiSquared_ / (2*numberOfPointUsedForFit_ - dimension); 
    }

    inline int getNDF() const {return (2*numberOfPointUsedForFit_ - dimension); }

    /// returns the point from which the track is passing by at the selected z
    inline GlobalPoint getTrackPoint(float z) const 
    {
      float delta_z = z - z0_;
      return GlobalPoint(
        track_params_vector_[TrackPar::x0] + track_params_vector_[TrackPar::tx] * delta_z,
        track_params_vector_[TrackPar::y0] + track_params_vector_[TrackPar::ty] * delta_z,
        z);
    }

    inline GlobalPoint getTrackCentrePoint()
    {
      return GlobalPoint(track_params_vector_[TrackPar::x0], track_params_vector_[TrackPar::y0], z0_);
    }

    AlgebraicSymMatrix22 trackPointInterpolationCovariance(float z) const;

    inline bool isValid() const { return valid_; }

    inline void setValid(bool valid) { valid_ = valid; }

    bool operator< (const CTPPSPixelLocalTrack &r);
    
  private:
    edm::DetSetVector<CTPPSPixelFittedRecHit> track_hits_vector_;

    /// track parameters: (x0, y0, tx, ty); x = x0 + tx*(z-z0) ...
    ParameterVector track_params_vector_;

    /// z where x0 and y0 are evaluated.
    /// filled from TotemRPGeometry::GetRPGlobalTranslation
    float z0_; 

    CovarianceMatrix par_covariance_matrix_;
  
    /// fit chi^2
    float chiSquared_;

    /// fit valid?
    bool valid_;

    int numberOfPointUsedForFit_;

};

#endif

