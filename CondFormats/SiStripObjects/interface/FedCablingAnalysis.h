#ifndef CondFormats_SiStripObjects_FedCablingAnalysis_H
#define CondFormats_SiStripObjects_FedCablingAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include <map>

class TH1;
class TProfile;

/** 
   @class FedCablingAnalysis
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FedCablingAnalysis : public CommissioningAnalysis {
  
 public:

  // ---------- Con(de)structors and typedefs ----------

  FedCablingAnalysis( const uint32_t& key );
  FedCablingAnalysis();
  virtual ~FedCablingAnalysis() {;}
  
  typedef std::map<uint32_t,uint16_t> Candidates;

  // ---------- Analysis results and histos ----------

  /** FED id. */
  inline const uint16_t& fedId() const;

  /** FED channel. */
  inline const uint16_t& fedCh() const; 

  /** Light level [ADC]. */
  inline const float& adcLevel() const;
  
  /** Container for candidate connections. */
  inline const Candidates& candidates() const;

  /** Pointer to FED id histogram. */
  inline const Histo& hFedId() const;

  /** Pointer to FED channel histogram. */
  inline const Histo& hFedCh() const;

  // ---------- Utility methods ----------
  
  /** Identifies if analysis is valid or not. */
  bool isValid() const;

  /** Prints analysis results. */
  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  // ---------- Private methods ----------
  
  /** Resets analysis member data. */
  void reset();

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& );

  /** Performs histogram anaysis. */
  void analyse();

  void algo1( TProfile*, TProfile* );
  void algo2( TProfile*, TProfile* );
  void algo3( TProfile*, TProfile* );
  
 private:

  // ---------- Private member data ----------

  /** FED id */
  uint16_t fedId_;

  /** FED channel */
  uint16_t fedCh_;

  /** Light level [ADC]. */
  float adcLevel_;

  /** Container for candidate connections. */
  Candidates candidates_;
  
  /** Threshold to identify candidate connections. */
  static const float threshold_;

  /** Histo containing FED id */
  Histo hFedId_;

  /** Histo containing FED channel */
  Histo hFedCh_;

};

// ---------- Inline methods ----------
  
const uint16_t& FedCablingAnalysis::fedId() const { return fedId_; }
const uint16_t& FedCablingAnalysis::fedCh() const { return fedCh_; } 
const float& FedCablingAnalysis::adcLevel() const { return adcLevel_; }
const FedCablingAnalysis::Candidates& FedCablingAnalysis::candidates() const { return candidates_; } 

const FedCablingAnalysis::Histo& FedCablingAnalysis::hFedId() const { return hFedId_; }
const FedCablingAnalysis::Histo& FedCablingAnalysis::hFedCh() const { return hFedCh_; }

#endif // CondFormats_SiStripObjects_FedCablingAnalysis_H

