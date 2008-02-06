#ifndef CondFormats_SiStripObjects_ApvLatencyAnalysis_H
#define CondFormats_SiStripObjects_ApvLatencyAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;
class TH1;

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

  inline const uint16_t& latency() const;
  
  inline const Histo& histo() const;

  void print( std::stringstream&, uint32_t not_used = 0 );

 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();
  
 private:

  /** APV latency setting */
  uint16_t latency_; 
  
  /** APV latency histo */
  Histo histo_;

 private:
  
  void deprecated();
  void analysis( const std::vector<const TProfile*>& histos, 
		 std::vector<unsigned short>& monitorables );
  
};

const uint16_t& ApvLatencyAnalysis::latency() const { return latency_; }
const ApvLatencyAnalysis::Histo& ApvLatencyAnalysis::histo() const { return histo_; }

#endif // CondFormats_SiStripObjects_ApvLatencyAnalysis_H

