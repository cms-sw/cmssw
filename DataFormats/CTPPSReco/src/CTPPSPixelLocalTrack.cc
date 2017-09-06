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

TMatrixD CTPPSPixelLocalTrack::trackPointInterpolationCovariance(double z) const
{
  TMatrixD h(2,4);
  h(0,0)=1;
  h(1,1)=1;
  h(0,2)=z-z0_;
  h(1,3)=z-z0_;
  
  TMatrixD cov_matr(dimension,dimension);
  for(int i=0; i<dimension; ++i)
    for(int j=0; j<dimension; ++j)
      cov_matr(i,j)=CovarianceMatrixElement(i,j);
  
  TMatrixD V_hT(cov_matr, TMatrixD::kMultTranspose, h);
  //h*=V_hT;
  //return h;
  return TMatrixD(h, TMatrixD::kMult, V_hT);
}

//----------------------------------------------------------------------------------------------------

CTPPSPixelLocalTrack::CTPPSPixelLocalTrack(double z0, const TVectorD & track_params_vector, 
      const TMatrixD &par_covariance_matrix, double chiSquared) 
      : z0_(z0), chiSquared_(chiSquared), valid_(true), numberOfPointUsedForFit_(0)
{
  for(int i=0; i<dimension; ++i)
  {
    track_params_vector_[i]=track_params_vector[i];
    for(int j=0; j<dimension; ++j)
    {
      CovarianceMatrixElement(i,j)=par_covariance_matrix(i,j);
    }
  }
}

//----------------------------------------------------------------------------------------------------

TVectorD CTPPSPixelLocalTrack::getParameterVector() const 
{
  TVectorD v(dimension);
  
  for (int i = 0; i < dimension; ++i)
    v[i] = track_params_vector_[i];
      
  return v;
}

//----------------------------------------------------------------------------------------------------

void CTPPSPixelLocalTrack::setParameterVector(const TVectorD & track_params_vector)
{
  for (int i = 0; i < dimension; ++i)
    track_params_vector_[i] = track_params_vector[i];
}

//----------------------------------------------------------------------------------------------------

TMatrixD CTPPSPixelLocalTrack::getCovarianceMatrix() const 
{
  TMatrixD m(dimension,dimension);
  
  for(int i=0; i<dimension; ++i)
    for(int j=0; j<dimension; ++j)
      m(i,j) = CovarianceMatrixElement(i,j);
      
  return m;
}

//----------------------------------------------------------------------------------------------------

void CTPPSPixelLocalTrack::setCovarianceMatrix(const TMatrixD &par_covariance_matrix)
{
  for(int i=0; i<dimension; ++i)
    for(int j=0; j<dimension; ++j)
      CovarianceMatrixElement(i,j) = par_covariance_matrix(i,j);
}

//----------------------------------------------------------------------------------------------------

bool operator< (const CTPPSPixelLocalTrack &l, const CTPPSPixelLocalTrack &r)
{
  if (l.z0_ < r.z0_)
    return true;
  if (l.z0_ > r.z0_)
    return false;
 
  for (int i = 0; i < CTPPSPixelLocalTrack::dimension; ++i)
  {
    if (l.track_params_vector_[i] < r.track_params_vector_[i])
      return true;
    if (l.track_params_vector_[i] > r.track_params_vector_[i])
      return false;
  }
 
  return false;
}

