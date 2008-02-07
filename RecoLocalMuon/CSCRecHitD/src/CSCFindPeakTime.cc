/* This is CSCFindPeakTime
 *
 * \author Dominique Fortin
 *
 * adapted from PulseTime.h originally written by S. Durkin
 */ 

#include <RecoLocalMuon/CSCRecHitD/interface/CSCFindPeakTime.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <cmath>
#include <iostream>

#include <vector>


/* FindPeakTime
 *
 */
bool CSCFindPeakTime::FindPeakTime( const int& tmax, const float* adc, float& t_zero, float& t_peak ) {
  
  // Initialize parameters in case fit fails
  float t0       = 0.;
  float N        = adc[1];
  t_peak         = 133.;
  float p0       = 4./t_peak;

  // If outside physical range, exit
  if ( tmax < 2 || tmax > 6 ) return false;

  float tb[4];
  for ( int time=0; time<4; ++time ){
    tb[time] = (tmax + time -1) * 50.;
  }

  int n_fit  = 4;
  if ( tmax == 6 ) n_fit = 3;

  float chi_min  = 1.e10;
  float chi_last = 1.e10;
  float tt0      = 0.;
  float chi2     = 0.;
  float del_t    = 100.;

  float x[4];
  float sx2 = 0.;
  float sxy = 0.;
  float NN = 0.;

  while ( del_t > 1. ) {
    sx2 = 0.;
    sxy = 0.;
        
    for ( int j=0; j < n_fit; ++j ) {
      x[j] = (tb[j] - tt0) * (tb[j] - tt0) * (tb[j] - tt0) * (tb[j] - tt0) * exp( -p0 * (tb[j] - tt0) );
      sx2  += x[j] * x[j];
      sxy  += x[j] * adc[j];
    }
    NN = sxy / sx2;
    
    // Compute chi^2
    chi2 = 0.0;
    for (int j=0; j < n_fit; ++j) chi2 += (adc[j] - NN * x[j]) * (adc[j] - NN * x[j]);

    // Test on chi^2 to decide what to do    
    if ( chi_last > chi2 ) {
      if (chi2 < chi_min ){
        t0      = tt0;
        N       = NN;
      }
      chi_last  = chi2;
      tt0       = tt0 + del_t;
    } else {
      tt0      = tt0 - 2. * del_t;
      del_t    = del_t / 2.;
      tt0      = tt0 + del_t;
      chi_last = 1.0e10;
    }
  }

  t_peak = t_peak;
  t_zero = tt0;

  return true;
}


/* FitCharge
 *
 */
void CSCFindPeakTime::FitCharge( const int& tmax, const float* adc, const float& t_zero, const float& t_peak, std::vector<float>& adcsFit ) {

  float p0  = 4./t_peak;
  float tt0 = t_zero;
  int n_fit = 4;
  if ( tmax == 6 ) n_fit=3;
  
  float tb[4], y[4];
  for ( int t = 0; t < 4; ++t ){
    tb[t] = (tmax + t - 1) * 50.;
    y[t] = adc[t];
  }

  // Find the normalization factor for the function
  float x[4];    
  float sx2 = 0.;
  float sxy = 0.;
  for ( int j=0; j < n_fit; ++j ) {
    float t = tb[j];
    x[j] = (t-tt0)*(t-tt0)*(t-tt0)*(t-tt0) * exp( -p0 * (t-tt0) );
    sx2  = sx2 + x[j] * x[j];
    sxy  = sxy + x[j] * y[j];
  }
  float N = sxy / sx2;
    

  // Now compute charge for a given t  --> only need charges at: tpeak-50, tpeak and tpeak+50
  for ( int i = 0; i < 3; ++i ) {
    float t = t_peak + (i - 1) * 50.;
    float q_fitted = N * (t-tt0)*(t-tt0)*(t-tt0)*(t-tt0) * exp( -p0 * (t-tt0) );
    adcsFit.push_back(q_fitted);
  }
  return;
}


