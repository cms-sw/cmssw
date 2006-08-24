#ifndef DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;

/**
   @class : ApvTimingAnalysis
   @author : M. Wingham, R.Bainbridge
   @brief : Histogram-based analysis for PLL-skew "monitorables". 
*/

class ApvTimingAnalysis : public CommissioningAnalysis {
  
 public:
  
  ApvTimingAnalysis() {;}
  virtual ~ApvTimingAnalysis() {;}

  /** Simple container class that holds various parameter values that
      are extracted from the "tick mark" histogram by the analysis. */
  class Monitorables : public CommissioningAnalysis::Monitorables {
  public:
    uint16_t pllCoarse_; // PLL coarse delay setting
    uint16_t pllFine_;   // PLL fine delay setting
    float delay_;        // Time delay (from coarse and fine values) [ns]
    float error_;        // Error on time delay [ns]
    float base_;         // Level of tick mark "base" [adc]
    float peak_;         // Level of tick mark "peak" [adc]
    float height_;       // Tick mark height [ADC]
    Monitorables() : 
      pllCoarse_(sistrip::invalid_), pllFine_(sistrip::invalid_),
      delay_(sistrip::invalid_), error_(sistrip::invalid_), 
      base_(sistrip::invalid_), peak_(sistrip::invalid_), 
      height_(sistrip::invalid_) {;}
    virtual ~Monitorables() {;}
    void print( std::stringstream& );
  };
  
  /** Takes a TProfile histo containing an APV tick mark, extracts
      various parameter values from the histogram, and fills the
      Monitorables object. */
  static void analysis( const TProfile* const, Monitorables& ); 
  static void deprecated( const TProfile* const, Monitorables& ) {;} 
  
 private:

  /** Deprecated method. */
  static void analysis( const std::vector<const TProfile*>& histos, 
			std::vector<unsigned short>& monitorables );
  
};

#endif // DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H

