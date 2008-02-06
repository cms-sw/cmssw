#ifndef CondFormats_SiStripObjects_FastFedCablingAnalysis_H
#define CondFormats_SiStripObjects_FastFedCablingAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include <map>

class TH1;
class TProfile;

/** 
   @class FastFedCablingAnalysis
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FastFedCablingAnalysis : public CommissioningAnalysis {
  
 public:

  // ---------- con(de)structors and typedefs ----------

  FastFedCablingAnalysis( const uint32_t& key);

  FastFedCablingAnalysis();

  virtual ~FastFedCablingAnalysis() {;}
  
  typedef std::map<uint32_t,uint16_t> Candidates;

  // ---------- public interface ----------

  /** Identifies if analysis is valid or not. */
  bool isValid() const;

  /** DCU hardware id (32-bits). */
  inline const uint32_t& dcuId() const;
  
  /** Linear Laser Driver channel. */
  inline const uint16_t& lldCh() const; 

  /** "High" light level [ADC]. */
  inline const float& highLevel() const;

  /** Spread in "high" ligh level [ADC]. */
  inline const float& highRms() const;

  /** "Low" light level [ADC]. */
  inline const float& lowLevel() const;

  /** Spread in "low" ligh level [ADC]. */
  inline const float& lowRms() const;

  /** Maximum light level in data [ADC]. */
  inline const float& max() const;
  
  /** Minimum light level in data [ADC]. */
  inline const float& min() const;

  /** Pointer to histogram. */
  inline const Histo& histo() const;

  // ---------- public print methods ----------

  /** Prints analysis results. */
  void print( std::stringstream&, uint32_t not_used = 0 );

  /** Overrides base method. */
  void header( std::stringstream& ) const;
  
  /** Overrides base method. */
  void summary( std::stringstream& ) const;
  
  // ---------- private methods ----------
  
 private:
  
  /** Resets analysis member data. */
  void reset();

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& );

  /** Performs histogram anaysis. */
  void analyse();

  // ---------- private member data ----------

 private:

  /** Extracted DCU id. */
  uint32_t dcuId_;

  /** Extracted LLD channel. */
  uint16_t lldCh_;

  /** */
  float highMedian_;

  /** */
  float highMean_;

  /** */
  float highRms_;

  /** */
  float lowMedian_;

  /** */
  float lowMean_;

  /** */
  float lowRms_;

  /** */
  float range_;

  /** */
  float midRange_;

  /** */
  float max_;

  /** */
  float min_;

  /** Threshold to identify digital high/low. */
  static const float threshold_;

  /** */
  static const uint16_t nBitsForDcuId_;

  /** */
  static const uint16_t nBitsForLldCh_;
  
  /** Histo */
  Histo histo_;

};

// ---------- Inline methods ----------
  
const uint32_t& FastFedCablingAnalysis::dcuId() const { return dcuId_; }
const uint16_t& FastFedCablingAnalysis::lldCh() const { return lldCh_; } 
const float& FastFedCablingAnalysis::highLevel() const { return highMean_; }
const float& FastFedCablingAnalysis::highRms() const { return highRms_; }
const float& FastFedCablingAnalysis::lowLevel() const { return lowMean_; }
const float& FastFedCablingAnalysis::lowRms() const { return lowRms_; }
const float& FastFedCablingAnalysis::max() const { return max_; }
const float& FastFedCablingAnalysis::min() const { return min_; }
const FastFedCablingAnalysis::Histo& FastFedCablingAnalysis::histo() const { return histo_; }

#endif // CondFormats_SiStripObjects_FastFedCablingAnalysis_H


