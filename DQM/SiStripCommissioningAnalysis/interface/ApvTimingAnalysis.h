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

  // ---------- Con(de)structors ----------
  
  ApvTimingAnalysis( const uint32_t& key );
  ApvTimingAnalysis();
  virtual ~ApvTimingAnalysis() {;}
  
  // ---------- Analysis results and histos ----------

  /** Time of tick mark rising edge [ns]. */
  inline const float& time() const; 

  /** Error on time of tick mark rising edge [ns]. */
  inline const float& error() const; 

  /** Optimum sampling point, defined w.r.t. rising edge [ns]. */
  inline const float& optimumSamplingPoint() const; 

  /** Sampling point of "reference" tick mark [ns]. */
  inline const float& refTime() const; 

  /** Static method to set global reference time [ns]. */
  void refTime( const float& time );
  
  /** Delay required to sync w.r.t. reference tick mark [ns]. */
  inline const float& delay() const; 

  /** Height of tick mark [ADC]. */
  inline const float& height() const;

  /** Baseline level of tick mark [ADC]. */
  inline const float& base() const; 

  /** Level of tick mark top [ADC]. */
  inline const float& peak() const; 

  /** Container of histogram pointer and title. */
  inline const Histo& histo() const;
  
  // ---------- Utility methods ----------
  
  /** Identifies if analysis is valid or not. */
  bool isValid();

  /** Prints analysis results. */
  void print( std::stringstream&, uint32_t not_used = 0 );
  
  /** Adds error codes for analysis (overrides private base). */ 
  inline void addErrorCode( const std::string& error );
  
 private:

  // ---------- Private methods ----------
  
  /** Resets analysis member data. */
  void reset();

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& );

  /** Performs histogram anaysis. */
  void analyse();
  
 private:

  // ---------- Private member data ----------
  
  /** Time of tick mark rising edge [ns]. */
  float time_;

  /** Error on time of tick mark rising edge [ns]. */
  float error_;

  /** Optimum sampling point, defined w.r.t. rising edge [ns]. */
  static const float optimumSamplingPoint_;

  /** Sampling point of "reference" tick mark [ns]. */
  static float refTime_;

  /** Delay required to sync w.r.t. reference tick mark [ns]. */
  float delay_;

  /** Height of tick mark [ADC]. */
  float height_;

  /** Baseline level of tick mark [ADC]. */
  float base_;

  /** Level of tick mark top [ADC]. */
  float peak_;

  /** Checks synchronization to ref time is done only once. */
  bool synchronized_;

  /** Container of histogram pointer and title. */
  Histo histo_;
  
};

// ---------- Inline methods ----------

const float& ApvTimingAnalysis::time() const { return time_; }
const float& ApvTimingAnalysis::error() const { return error_; }
const float& ApvTimingAnalysis::optimumSamplingPoint() const { return optimumSamplingPoint_; }
const float& ApvTimingAnalysis::refTime() const { return refTime_; }
const float& ApvTimingAnalysis::delay() const { return delay_; }
const float& ApvTimingAnalysis::height() const { return height_; }
const float& ApvTimingAnalysis::base() const { return base_; }
const float& ApvTimingAnalysis::peak() const { return peak_; }
const ApvTimingAnalysis::Histo& ApvTimingAnalysis::histo() const { return histo_; }
void ApvTimingAnalysis::addErrorCode( const std::string& error ) { CommissioningAnalysis::addErrorCode(error) ;}

#endif // DQM_SiStripCommissioningAnalysis_ApvTimingAnalysis_H



