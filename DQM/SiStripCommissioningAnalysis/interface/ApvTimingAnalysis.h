#ifndef DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;

/**
   @class : ApvTimingAnalysis
   @author : M. Wingham
   @brief : Histogram-based analysis for PLL-skew "monitorables". 
*/

class ApvTimingAnalysis : public CommissioningAnalysis {
  
 public:
  
  ApvTimingAnalysis() {;}
  virtual ~ApvTimingAnalysis() {;}

  /** Simple container class that holds various parameter values that
      are extracted from the "tick mark" histogram by the analysis. */
  class Monitorables {
  public:
    uint16_t coarse_; // PLL coarse delay setting
    uint16_t fine_;   // PLL fine delay setting
    float delay_;     // Time delay (from coarse and fine values) [ns]
    float error_;     // Error on time delay [ns]
    float base_;      // Level of tick mark "base" [adc]
    float peak_;      // Level of tick mark "peak" [adc]
    float height_;    // Tick mark height [ADC]
    Monitorables() : 
      coarse_(0), fine_(0), 
      delay_(0.), error_(0.), 
      base_(0.), peak_(0.), height_(0.) {;}
    void print( std::stringstream& );
  };
  
  /** Takes a TProfile histo containing an APV tick mark, extracts
      various parameter values from the histogram, and fills the
      Monitorables object. */
  static void analysis( const TProfile* const, Monitorables& ); 
  
  /** Takes a vector containing one TProfile of a tick mark and fills a vector of 2 unsigned shorts representing the rise time of the tick. The first is a coarse-time measurement (units of 25ns), the second a fine-time (units 1/24th of the coarse). */
  static void analysis( const std::vector<const TProfile*>& histos, 
			std::vector<unsigned short>& monitorables );
  
};

#endif // DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H

