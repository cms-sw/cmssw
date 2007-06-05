#ifndef DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TH1;

/**
   @class ApvTimingAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Analysis for timing run using APV tick marks.
*/

class ApvTimingAnalysis : public CommissioningAnalysis {
  
 public:
  
  ApvTimingAnalysis( const uint32_t& key );
  ApvTimingAnalysis();
  virtual ~ApvTimingAnalysis() {;}
  
  // ---------- Access to analysis and histos ----------

  inline const float& time() const; 
  inline const float& maxTime() const; 
  inline const float& delay() const; 
  inline const float& error() const; 
  inline const float& base() const; 
  inline const float& peak() const; 
  inline const float& height() const;
  
  inline const Histo& histo() const;
  
  void maxTime( const float& time ); //@@ make this and data static!

  // ---------- Utility methods ----------
  
  bool isValid();

  void print( std::stringstream&, uint32_t not_used = 0 );
  
  /** Override base method for public access. */
  inline void addErrorCode( const std::string& error );
  
 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();
  
 private:
  
  /** Time of tick mark rising edge [ns] */
  float time_;

  /** Time of sampling point of last tick mark [ns] */
  float maxTime_;

  /** Delay required to synchronise */ 
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
  
  /** */
  float optimumSamplingPoint_;
  
};

const float& ApvTimingAnalysis::time() const { return time_; }
const float& ApvTimingAnalysis::maxTime() const { return maxTime_; }
const float& ApvTimingAnalysis::delay() const { return delay_; }
const float& ApvTimingAnalysis::error() const { return error_; }
const float& ApvTimingAnalysis::base() const { return base_; }
const float& ApvTimingAnalysis::peak() const { return peak_; }
const float& ApvTimingAnalysis::height() const { return height_; }
const ApvTimingAnalysis::Histo& ApvTimingAnalysis::histo() const { return histo_; }
void ApvTimingAnalysis::addErrorCode( const std::string& error ) { CommissioningAnalysis::addErrorCode(error) ;}

#endif // DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H



