// This is CSCFindPeakTime

#include <RecoLocalMuon/CSCRecHitD/src/CSCFindPeakTime.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <cmath>

CSCFindPeakTime::CSCFindPeakTime( const edm::ParameterSet& ps ): 
  useAverageTime(false), useParabolaFit(false), useFivePoleFit(false) {

  useAverageTime = ps.getParameter<bool>("UseAverageTime");
  useParabolaFit = ps.getParameter<bool>("UseParabolaFit");
  useFivePoleFit = ps.getParameter<bool>("UseFivePoleFit");
  LogTrace("CSCRecHit") << "[CSCFindPeakTime] useAverageTime=" << useAverageTime <<
    ", useParabolaFit=" << useParabolaFit << ", useFivePoleFit=" << useFivePoleFit;
}

float CSCFindPeakTime::peakTime( int tmax, const float* adc, float t_peak){
  if ( useAverageTime ) {
    return averageTime( tmax, adc );
  }
  else if ( useParabolaFit ) {
    return parabolaFitTime( tmax, adc );
  }
  else if ( useFivePoleFit ) {
     return fivePoleFitTime( tmax, adc, t_peak);
  }
  else {
  // return something, anyway.. may as well be average
    return averageTime( tmax, adc );
  }
}

float CSCFindPeakTime::averageTime( int tmax, const float* adc ) {
   float sum  = 0.;
   float sumt = 0.;
   for (size_t i=0; i<4; ++i){
     sum  += adc[i];
     sumt += adc[i] * ( tmax - 1 + i );

   }
   return sumt/sum * 50.; //@@ in ns. May be some bin width offset things to handle here?
}

float CSCFindPeakTime::parabolaFitTime( int tmax, const float* adc ) {
  // 3-point parabolic fit, from Andy Kubik
 
  // We calculate offset to tmax by finding the peak of a parabola through three points
   float tpeak = tmax;
   float tcorr = 0;

   // By construction, input array adc is for bins tmax-1 to tmax+2
   float y1 = adc[0];
   float y2 = adc[1];
   float y3 = adc[2];

   // Checked and simplified... Tim Cox 08-Apr-2009
   // Denominator is not zero unless we fed in nonsense values with y2 not the peak!
   if ( (y1+y3) < 2.*y2 ) tcorr =  0.5 * ( y1 - y3 ) / ( y1 - 2.*y2 + y3 );
   tpeak += tcorr;

   LogTrace("CSCFindPeakTime") << "[CSCFindPeakTime] tmax=" << tmax 
     << ", parabolic peak time is tmax+" << tcorr <<" bins, or " << tpeak*50. << " ns";
   
   return tpeak * 50.; // convert to ns.
}

float CSCFindPeakTime::fivePoleFitTime( int tmax, const float* adc, float t_peak ) {

  // Input is 
  // tmax   = bin# 0-7 containing max SCA pulse height  
  // adc    = 4-dim array containing SCA pulse heights in bins tmax-1 to tmax+2
  // t_peak = input estimate for SCA peak time

  // Returned value is improved (we hope) estimate for SCA peak time

  // Algorithm is to fit five-pole Semi-Gaussian function for start time of SCA pulse, t0
  // (The SCA peak is assumed to be 133 ns from t0.)
  // Note that t^4 in time domain corresponds to 1/t^5 in frequency domain (that's the 5 poles).

  // Initialize parameters to sensible (?) values

  float t0       = 0.;
  float t0peak   = 133.;   // this is offset of peak from start time t0
  float p0       = 4./t0peak;

  // Require that tmax is in range 2-6 of bins the eight SCA time bins 0-7
  // (Bins 0, 1 used for dynamic ped)

  if ( tmax < 2 || tmax > 6 ) return t_peak; //@@ Just return the input value

  // Set up time bins to match adc[4] input

  float tb[4];
  for ( int time=0; time<4; ++time ){
    tb[time] = (tmax + time -1) * 50.;
  }

  // How many time bins are we fitting?

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
  float fN  = 0.;

  while ( del_t > 1. ) {
    sx2 = 0.;
    sxy = 0.;
        
    for ( int j=0; j < n_fit; ++j ) {
      float tdif = tb[j] - tt0;
      x[j] = tdif * tdif * tdif * tdif * exp( -p0 * tdif );
      sx2 += x[j] * x[j];
      sxy += x[j] * adc[j];
    }
    fN = sxy / sx2; // least squares fit over time bins i to adc[i] = fN * fivePoleFunction[i]
    
    // Compute chi^2
    chi2 = 0.0;
    for (int j=0; j < n_fit; ++j) chi2 += (adc[j] - fN * x[j]) * (adc[j] - fN * x[j]);

    // Test on chi^2 to decide what to do    
    if ( chi_last > chi2 ) {
      if (chi2 < chi_min ){
        t0      = tt0;
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

  return t0 + t0peak;
}



void CSCFindPeakTime::fivePoleFitCharge( int tmax, const float* adc, const float& t_zero, const float& t_peak, std::vector<float>& adcsFit ) {

  //@@ This code can certainly be replaced by fivePoleFitTime above, but I haven't time to do that now (Tim).

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
    

  // Now compute charge for a given t  --> only need charges at: t_peak-50, t_peak and t_peak+50
  for ( int i = 0; i < 3; ++i ) {
    float t = t_peak + (i - 1) * 50.;
    float q_fitted = N * (t-tt0)*(t-tt0)*(t-tt0)*(t-tt0) * exp( -p0 * (t-tt0) );
    adcsFit.push_back(q_fitted);
  }
  return;
}


