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

AlgebraicSymMatrix22 CTPPSPixelLocalTrack::trackPointInterpolationCovariance(float z) const
{
  math::Matrix<2, dimension>::type h;
  h(0,0)=1;
  h(1,1)=1;
  h(0,2)=z-z0_;
  h(1,3)=z-z0_;
  
  CovarianceMatrix cov_matr;
  for(unsigned int i=0; i<dimension; ++i)
    for(unsigned int j=0; j<dimension; ++j)
      cov_matr[i][j]=covarianceMatrixElement(i,j);

  return ROOT::Math::Similarity(h,cov_matr);
  
}

//----------------------------------------------------------------------------------------------------

CTPPSPixelLocalTrack::CTPPSPixelLocalTrack(float z0, const ParameterVector & track_params_vector, 
      const CovarianceMatrix &par_covariance_matrix, float chiSquared) 
      : z0_(z0), chiSquared_(chiSquared), valid_(true), numberOfPointUsedForFit_(0)
{
  for(unsigned int i=0; i<dimension; ++i)
  {
    track_params_vector_[i]=track_params_vector[i];
    for(unsigned int j=0; j<dimension; ++j)
    {
      covarianceMatrixElement(i,j)=par_covariance_matrix(i,j);
    }
  }
}

//----------------------------------------------------------------------------------------------------

CTPPSPixelLocalTrack::ParameterVector CTPPSPixelLocalTrack::getParameterVector() const 
{
  ParameterVector v;
  
  for (unsigned int i = 0; i < dimension; ++i)
    v[i] = track_params_vector_[i];
      
  return v;
}

//----------------------------------------------------------------------------------------------------

CTPPSPixelLocalTrack::CovarianceMatrix CTPPSPixelLocalTrack::getCovarianceMatrix() const 
{
  CovarianceMatrix m;
  
  for(int i=0; i<dimension; ++i)
    for(int j=0; j<dimension; ++j)
      m(i,j) = covarianceMatrixElement(i,j);
      
  return m;
}

//----------------------------------------------------------------------------------------------------

bool CTPPSPixelLocalTrack::operator< (const CTPPSPixelLocalTrack &r)
{
  if (z0_ < r.z0_)
    return true;
  if (z0_ > r.z0_)
    return false;
 
  for (int i = 0; i < CTPPSPixelLocalTrack::dimension; ++i)
  {
    if (track_params_vector_[i] < r.track_params_vector_[i])
      return true;
    if (track_params_vector_[i] > r.track_params_vector_[i])
      return false;
  }
 
  return false;

}

