#ifndef DQM_SiStripCommissioningAnalysis_ApvLatencyAnalysis_H
#define DQM_SiStripCommissioningAnalysis_ApvLatencyAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <vector>

class TProfile;

/** 
   \class ApvLatencyAnalysis
   \brief Concrete implementation of the histogram-based analysis for
   ApvLatency "monitorables". 
*/
class ApvLatencyAnalysis : public CommissioningAnalysis {
  
 public:

  ApvLatencyAnalysis() {;}
  virtual ~ApvLatencyAnalysis() {;}

  /** Simple container class that holds various parameter values that
      are extracted from the "tick mark" histogram by the analysis. */
  class Monitorables : public CommissioningAnalysis::Monitorables {
  public:
    uint16_t apvLatency_; // APV latency setting
    Monitorables() : 
      apvLatency_(sistrip::invalid_) {;}
    virtual ~Monitorables() {;}
    void print( std::stringstream& );
  };

  /** Takes a vector containing one TH1F of a scan through coarse
      (25ns) trigger latency settings, filled with the number of
      recorded hits per setting. The monitorables vector is filled
      with the latency for the largest number of recorded hits over
      5*sigma of the noise. */
  static void analysis( const std::vector<const TProfile*>& histos, 
			std::vector<unsigned short>& monitorables );
  
};

#endif // DQM_SiStripCommissioningAnalysis_ApvLatencyAnalysis_H

