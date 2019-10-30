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

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrackRecoInfo.h"
//----------------------------------------------------------------------------------------------------

class CTPPSPixelFittedRecHit : public CTPPSPixelRecHit {
public:
  CTPPSPixelFittedRecHit(const CTPPSPixelRecHit& hit,
                         const GlobalPoint& space_point_on_det,
                         const LocalPoint& residual,
                         const LocalPoint& pull)
      : CTPPSPixelRecHit(hit),
        space_point_on_det_(space_point_on_det),
        residual_(residual),
        pull_(pull),
        isUsedForFit_(false),
        isRealHit_(false) {}

  CTPPSPixelFittedRecHit()
      : CTPPSPixelRecHit(),
        residual_(LocalPoint(0, 0)),
        pull_(LocalPoint(0, 0)),
        isUsedForFit_(false),
        isRealHit_(false) {}

  virtual ~CTPPSPixelFittedRecHit() {}

  inline const GlobalPoint& globalCoordinates() const { return space_point_on_det_; }
  inline float xResidual() const { return residual_.x(); }
  inline float yResidual() const { return residual_.y(); }

  inline float xPull() const { return pull_.x(); }
  inline float yPull() const { return pull_.y(); }

  inline float xPullNormalization() const { return residual_.x() / pull_.x(); }
  inline float yPullNormalization() const { return residual_.y() / pull_.y(); }

  inline void setIsUsedForFit(bool usedForFit) {
    if (usedForFit)
      isRealHit_ = true;
    isUsedForFit_ = usedForFit;
  }
  inline bool isUsedForFit() const { return isUsedForFit_; }

  inline void setIsRealHit(bool realHit) {
    if (!realHit)
      isUsedForFit_ = false;
    isRealHit_ = realHit;
  }
  inline bool isRealHit() const { return isRealHit_; }

private:
  GlobalPoint space_point_on_det_;  ///< mm
  LocalPoint residual_;             ///< mm
  LocalPoint pull_;                 ///< normalised residual
  bool isUsedForFit_;
  bool isRealHit_;
};

class CTPPSPixelLocalTrack {
public:
  enum class TrackPar { x0 = 0, y0 = 1, tx = 2, ty = 3 };

  ///< parameter vector size
  static constexpr int dimension = 4;
  typedef math::Error<dimension>::type CovarianceMatrix;
  typedef math::Vector<dimension>::type ParameterVector;

  ///< covariance matrix size
  static constexpr int covarianceSize = dimension * dimension;

  CTPPSPixelLocalTrack()
      : z0_(0),
        chiSquared_(0),
        valid_(false),
        numberOfPointsUsedForFit_(0),
        recoInfo_(CTPPSpixelLocalTrackReconstructionInfo::invalid) {}

  CTPPSPixelLocalTrack(float z0,
                       const ParameterVector& track_params_vector,
                       const CovarianceMatrix& par_covariance_matrix,
                       float chiSquared);

  ~CTPPSPixelLocalTrack() {}

  inline const edm::DetSetVector<CTPPSPixelFittedRecHit>& hits() const { return track_hits_vector_; }
  inline void addHit(unsigned int detId, const CTPPSPixelFittedRecHit& hit) {
    track_hits_vector_.find_or_insert(detId).push_back(hit);
    if (hit.isUsedForFit())
      ++numberOfPointsUsedForFit_;
  }

  inline float x0() const { return track_params_vector_[(unsigned int)TrackPar::x0]; }
  inline float x0Sigma() const {
    return sqrt(par_covariance_matrix_[(unsigned int)TrackPar::x0][(unsigned int)TrackPar::x0]);
  }
  inline float x0Variance() const {
    return par_covariance_matrix_[(unsigned int)TrackPar::x0][(unsigned int)TrackPar::x0];
  }

  inline float y0() const { return track_params_vector_[(unsigned int)TrackPar::y0]; }
  inline float y0Sigma() const {
    return sqrt(par_covariance_matrix_[(unsigned int)TrackPar::y0][(unsigned int)TrackPar::y0]);
  }
  inline float y0Variance() const {
    return par_covariance_matrix_[(unsigned int)TrackPar::y0][(unsigned int)TrackPar::y0];
  }

  inline float z0() const { return z0_; }
  inline void setZ0(float z0) { z0_ = z0; }

  inline float tx() const { return track_params_vector_[(unsigned int)TrackPar::tx]; }
  inline float txSigma() const {
    return sqrt(par_covariance_matrix_[(unsigned int)TrackPar::tx][(unsigned int)TrackPar::tx]);
  }

  inline float ty() const { return track_params_vector_[(unsigned int)TrackPar::ty]; }
  inline float tySigma() const {
    return sqrt(par_covariance_matrix_[(unsigned int)TrackPar::ty][(unsigned int)TrackPar::ty]);
  }

  inline GlobalVector directionVector() const {
    GlobalVector vect(tx(), ty(), 1);
    return vect.unit();
  }

  inline const ParameterVector& parameterVector() const { return track_params_vector_; }

  inline const CovarianceMatrix& covarianceMatrix() const { return par_covariance_matrix_; }

  inline float chiSquared() const { return chiSquared_; }

  inline float chiSquaredOverNDF() const {
    if (numberOfPointsUsedForFit_ <= dimension / 2)
      return -999.;
    else
      return chiSquared_ / (2 * numberOfPointsUsedForFit_ - dimension);
  }

  inline int ndf() const { return (2 * numberOfPointsUsedForFit_ - dimension); }

  /// returns the point from which the track is passing by at the selected z
  inline GlobalPoint trackPoint(float z) const {
    float delta_z = z - z0_;
    return GlobalPoint(
        track_params_vector_[(unsigned int)TrackPar::x0] + track_params_vector_[(unsigned int)TrackPar::tx] * delta_z,
        track_params_vector_[(unsigned int)TrackPar::y0] + track_params_vector_[(unsigned int)TrackPar::ty] * delta_z,
        z);
  }

  inline GlobalPoint trackCentrePoint() {
    return GlobalPoint(
        track_params_vector_[(unsigned int)TrackPar::x0], track_params_vector_[(unsigned int)TrackPar::y0], z0_);
  }

  AlgebraicSymMatrix22 trackPointInterpolationCovariance(float z) const;

  inline bool isValid() const { return valid_; }

  inline void setValid(bool valid) { valid_ = valid; }

  bool operator<(const CTPPSPixelLocalTrack& r);

  inline CTPPSpixelLocalTrackReconstructionInfo recoInfo() const { return recoInfo_; }
  inline void setRecoInfo(CTPPSpixelLocalTrackReconstructionInfo recoInfo) { recoInfo_ = recoInfo; }

  inline unsigned short numberOfPointsUsedForFit() const { return numberOfPointsUsedForFit_; }

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

  /// number of points used for the track fit
  int numberOfPointsUsedForFit_;

  CTPPSpixelLocalTrackReconstructionInfo recoInfo_;
};

#endif
