#ifndef DQM_SiStripCommissioningAnalysis_FedCablingAnalysis_H
#define DQM_SiStripCommissioningAnalysis_FedCablingAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;

/** 
   @class FedCablingAnalysis
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FedCablingAnalysis : public CommissioningAnalysis {
  
 public:
  
  FedCablingAnalysis( const uint32_t& key );
  FedCablingAnalysis();
  virtual ~FedCablingAnalysis() {;}
  
  inline const uint16_t& fedId() const;
  inline const uint16_t& fedCh() const; 
  inline const float& level() const;
  inline const uint16_t& num() const; 

  inline const Histo& hFedId() const;
  inline const Histo& hFedCh() const;
  
  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  void reset();
  void extract( const std::vector<TProfile*>& );
  void analyse();

 private:
  
  /** FED id */
  uint16_t fedId_;
  /** FED channel */
  uint16_t fedCh_;
  /** Signal level [adc] */
  float    level_;
  /** Number of candidates for connection */
  uint16_t num_;
  
  /** Histo containing FED id */
  Histo hFedId_;
  /** Histo containing FED channel */
  Histo hFedCh_;

};
  
const uint16_t& FedCablingAnalysis::fedId() const { return fedId_; }
const uint16_t& FedCablingAnalysis::fedCh() const { return fedCh_; } 
const float& FedCablingAnalysis::level() const { return level_; }
const uint16_t& FedCablingAnalysis::num() const { return num_; } 

const FedCablingAnalysis::Histo& FedCablingAnalysis::hFedId() const { return hFedId_; }
const FedCablingAnalysis::Histo& FedCablingAnalysis::hFedCh() const { return hFedCh_; }

#endif // DQM_SiStripCommissioningAnalysis_FedCablingAnalysis_H

