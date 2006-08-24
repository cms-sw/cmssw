#ifndef DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H
#define DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;

/** 
   @class : VpspScanAnalysis
   @author : M. Wingham, R.Bainbridge
   @brief : Histogram-based analysis for VPSP "monitorables". 
*/
class VpspScanAnalysis : public CommissioningAnalysis {
  
 public:

  VpspScanAnalysis() {;}
  virtual ~VpspScanAnalysis() {;}

  class TProfiles {
  public:
    TProfile* vpsp0_; // "VPSP scan" histo for APV0 of pair
    TProfile* vpsp1_; // "VPSP scan" histo for APV1 of pair
    TProfiles() : 
      vpsp0_(0), vpsp1_(0) {;}
    ~TProfiles() {;}
    void print( std::stringstream& );
  };
  
  /** Simple container class that holds various parameter values that
      are extracted from the "VPSP scan" histogram by the analysis. */
  class Monitorables : public CommissioningAnalysis::Monitorables {
  public:
    uint16_t vpsp0_; // VPSP setting for APV0
    uint16_t vpsp1_; // VPSP setting for APV1
    Monitorables() : 
      vpsp0_(sistrip::invalid_), 
      vpsp1_(sistrip::invalid_) {;}
    virtual ~Monitorables() {;}
    void print( std::stringstream& );
  };
  
  /** */
  static void analysis(const TProfiles&, Monitorables& );
  static void deprecated(const TProfiles&, Monitorables& );
  
 private:
  
  /** Deprecated. */
  static void analysis(const std::vector<const TProfile*>& histos, 
		       std::vector<unsigned short>& monitorables);
  
};

#endif // DQM_SiStripCommissioningAnalysis_VpspScanAnalysis_H

