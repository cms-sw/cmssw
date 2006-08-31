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
   @author : R.Bainbridge
   @brief : Histogram-based analysis for FED cabling
*/
class FedCablingAnalysis : public CommissioningAnalysis {
  
 public:

  FedCablingAnalysis() {;}
  virtual ~FedCablingAnalysis() {;}
  
  class TProfiles {
  public:
    TProfile* fedId_; // Histo containing FED id
    TProfile* fedCh_; // Histo containing FED channel
    TProfiles() : 
      fedId_(0), fedCh_(0) {;}
    ~TProfiles() {;}
    void print( std::stringstream& );
  };
  
  /** Simple container class that holds various parameter values that
      are extracted from the histogram(s) by the analysis. */
  class Monitorables : public CommissioningAnalysis::Monitorables {
  public:
    uint16_t fedId_; // FED id 
    uint16_t fedCh_; // FED channel
    float    level_; // Signal level [adc]
    uint16_t num_;   // Number of candidates for connection
    Monitorables() : 
      fedId_(sistrip::invalid_), 
      fedCh_(sistrip::invalid_),
      level_(sistrip::invalid_),
      num_(sistrip::invalid_) {;}
    virtual ~Monitorables() {;}
    void print( std::stringstream& );
  };
  
  /** Takes a TProfile histo containing an APV tick mark, extracts
      various parameter values from the histogram, and fills the
      Monitorables object. */
  static void analysis( const TProfiles&, Monitorables& ); 
  
};

#endif // DQM_SiStripCommissioningAnalysis_FedCablingAnalysis_H

