#ifndef DQM_SiStripCommissioningAnalysis_FedTimingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_FedTimingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;

/**
   @class FedTimingAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Analysis for timing run using APV tick marks.
*/

class FedTimingAnalysis : public CommissioningAnalysis {
  
 public:
  
  FedTimingAnalysis();
  virtual ~FedTimingAnalysis() {;}
  
  inline const uint16_t& pllCoarse() const;
  inline const uint16_t& pllFine() const; 
  inline const float& delay() const; 
  inline const float& error() const; 
  inline const float& base() const; 
  inline const float& peak() const; 
  inline const float& height() const;

  inline const Histo& histo() const;

  void print( std::stringstream&, uint32_t not_used = 0 );

 private:
  
  void reset();
  void extract( const std::vector<TProfile*>& );
  void analyse();
  
 private:

  /** PLL coarse delay setting */
  uint16_t pllCoarse_; 
  /** PLL fine delay setting */
  uint16_t pllFine_;
  /** Timing delay [ns] */
  float delay_;
  /** Error on time delay [ns] */
  float error_;
  /** Level of tick mark "base" [adc] */
  float base_;
  /** Level of tick mark "peak" [adc] */
  float peak_;
  /** Tick mark height [adc] */
  float height_;
  
  /** APV tick mark */
  Histo histo_;
    
  
};

const uint16_t& FedTimingAnalysis::pllCoarse() const { return pllCoarse_; } 
const uint16_t& FedTimingAnalysis::pllFine() const { return pllFine_; }
const float& FedTimingAnalysis::delay() const { return delay_; }
const float& FedTimingAnalysis::error() const { return error_; }
const float& FedTimingAnalysis::base() const { return base_; }
const float& FedTimingAnalysis::peak() const { return peak_; }
const float& FedTimingAnalysis::height() const { return height_; }
const FedTimingAnalysis::Histo& FedTimingAnalysis::histo() const { return histo_; }

#endif // DQM_SiStripCommissioningAnalysis_FedTimingAnalysis_H



