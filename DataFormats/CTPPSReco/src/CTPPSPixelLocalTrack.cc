/*
 *
* This is a part of CTPPS offline software.
* Author:
*   Fabrizio Ferro (ferro@ge.infn.it)
*   Enrico Robutti (robutti@ge.infn.it)
*   Fabio Ravera   (fabio.ravera@cern.ch)
*
*/

#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

//----------------------------------------------------------------------------------------------------

AlgebraicSymMatrix22 CTPPSPixelLocalTrack::trackPointInterpolationCovariance(float z) const {
  math::Matrix<2, dimension>::type h;
  h(0, 0) = 1;
  h(1, 1) = 1;
  h(0, 2) = z - z0_;
  h(1, 3) = z - z0_;

  return ROOT::Math::Similarity(h, par_covariance_matrix_);
}

//----------------------------------------------------------------------------------------------------

CTPPSPixelLocalTrack::CTPPSPixelLocalTrack(float z0,
                                           const ParameterVector &track_params_vector,
                                           const CovarianceMatrix &par_covariance_matrix,
                                           float chiSquared)
    : track_params_vector_(track_params_vector),
      z0_(z0),
      par_covariance_matrix_(par_covariance_matrix),
      chiSquared_(chiSquared),
      valid_(true),
      numberOfPointsUsedForFit_(0),
      recoInfo_(CTPPSpixelLocalTrackReconstructionInfo::invalid) {}

bool CTPPSPixelLocalTrack::operator<(const CTPPSPixelLocalTrack &r) {
  if (z0_ < r.z0_)
    return true;
  if (z0_ > r.z0_)
    return false;

  for (int i = 0; i < CTPPSPixelLocalTrack::dimension; ++i) {
    if (track_params_vector_[i] < r.track_params_vector_[i])
      return true;
    if (track_params_vector_[i] > r.track_params_vector_[i])
      return false;
  }

  return false;
}
