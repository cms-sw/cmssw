#ifndef DQM_SiStripCommissioningAnalysis_FastFedCablingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_FastFedCablingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
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

  // ---------- Con(de)structors and typedefs ----------

  FastFedCablingAnalysis( const uint32_t& key);
  FastFedCablingAnalysis();
  virtual ~FastFedCablingAnalysis() {;}
  
  typedef std::map<uint32_t,uint16_t> Candidates;

  // ---------- Analysis results and histos ----------

  /** FED id. */
  inline const uint32_t& dcuId() const;
  
  /** FED channel. */
  inline const uint16_t& lldCh() const; 

  /** Pointer to histogram. */
  inline const Histo& histo() const;

  // ---------- Utility methods ----------
  
  /** Identifies if analysis is valid or not. */
  bool isValid() const;

  /** Prints analysis results. */
  void print( std::stringstream&, uint32_t not_used = 0 );

 protected:
  
  /** Overrides base method. */
  void header( std::stringstream& ) const;
  
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

const FastFedCablingAnalysis::Histo& FastFedCablingAnalysis::histo() const { return histo_; }

#endif // DQM_SiStripCommissioningAnalysis_FastFedCablingAnalysis_H

