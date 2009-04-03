/** This is CSCFindPeakTime
 *
 *  Used to provide improved estimate of SCA peak time.
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>


class CSCFindPeakTime
{
 public:
  
  explicit CSCFindPeakTime( const edm::ParameterSet& ps );
  
  ~CSCFindPeakTime(){}; 
  
  /// Basic result of this class
  float peakTime( int tmax, const float* adc, float t_zero, float t_peak);

  /// Weighted avergae of time bins
  float averageTime( int tmax, const float* adc );

  /// Parabolic fit to three time bins centered on maximum
  float parabolaFitTime( int tmax, const float* adc );

  /**
   * Based on RecoLocalMuon/CSCStandAlone/interface/PulseTime.h  by S. Durkin,
   * and ported by D. Fortin. Comments updated by Tim Cox Apr 2009.
   *
   * The SCA pulse shape should be representable by a function <BR>
   *    N*(p0^2/256/exp(-4)) * (t-t0)^4 * exp( -p0*(t-t0) )
   *	
   * Rather than do a full fit with varying peak time too, assume the
   * peak time is fixed to 133 nsec w.r.t. start time, t0, and fit for t0.
   * The fit uses a binary search in t0, and at each step calculates the overall normalization factor
   * between the function and the SCA pulse height as a least-squares fit over the 4 time bins
   * tmax -1, tmax, tmax+1, tmax+2
   *	
   * Note: tpeak=4/p0 (nsec) and adc[0] is arbitrarily defined a time of 0.0 nsec. 
   *
   */
  bool fourPoleFitTime( int tmax, const float* adc, float t_zero, float t_peak ); 

  /**
   * Integrated charge after fourPoleFitTime
   */
  //@@ Needs work and interface fixes!!
  void fourPoleFitCharge( int tmax, const float* adc, const float& t_zero, const float& t_peak, std::vector<float>& adcsFit );
  
 private:

  bool useAverageTime;
  bool useParabolaFit;
  bool useFourPoleFit;  
  
};
