#ifndef CondFormats_SiStripObjects_ApvLatencyAnalysis_H
#define CondFormats_SiStripObjects_ApvLatencyAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

/** 
   @class ApvLatencyAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Analysis for APV latency scan.
*/
class ApvLatencyAnalysis : public CommissioningAnalysis {
  
 public:

  ApvLatencyAnalysis(  const uint32_t& key );

  ApvLatencyAnalysis();

  virtual ~ApvLatencyAnalysis() {;}

  friend class ApvLatencyAlgorithm;

  inline const uint16_t& latency() const;

  void print( std::stringstream&, uint32_t not_used = 0 );
  
  void reset();
  
 private:

  /** APV latency setting */
  uint16_t latency_; 
  
};

const uint16_t& ApvLatencyAnalysis::latency() const { return latency_; }

#endif // CondFormats_SiStripObjects_ApvLatencyAnalysis_H

