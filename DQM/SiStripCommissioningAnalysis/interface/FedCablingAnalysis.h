#ifndef DQM_SiStripCommissioningAnalysis_FedCablingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_FedCablingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>
#include <map>

class TH1;

/** 
   @class FedCablingAnalysis
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FedCablingAnalysis : public CommissioningAnalysis {
  
 public:

  typedef std::map<uint32_t,uint16_t> Candidates;

  FedCablingAnalysis( const uint32_t& key );
  FedCablingAnalysis();
  virtual ~FedCablingAnalysis() {;}

  // Connection info
  inline const uint16_t& fedId() const;
  inline const uint16_t& fedCh() const; 
  uint16_t adcLevel() const;
  inline const Candidates& candidates() const;
  
  inline const Histo& hFedId() const;
  inline const Histo& hFedCh() const;

  bool isValid();

  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();

 private:
  
  /** FED id */
  uint16_t fedId_;
  /** FED channel */
  uint16_t fedCh_;
  /** Number of candidates for connection */
  Candidates candidates_;
  
  /** Histo containing FED id */
  Histo hFedId_;
  /** Histo containing FED channel */
  Histo hFedCh_;

};
  
const uint16_t& FedCablingAnalysis::fedId() const { return fedId_; }
const uint16_t& FedCablingAnalysis::fedCh() const { return fedCh_; } 
const FedCablingAnalysis::Candidates& FedCablingAnalysis::candidates() const { return candidates_; } 

const FedCablingAnalysis::Histo& FedCablingAnalysis::hFedId() const { return hFedId_; }
const FedCablingAnalysis::Histo& FedCablingAnalysis::hFedCh() const { return hFedCh_; }

#endif // DQM_SiStripCommissioningAnalysis_FedCablingAnalysis_H

