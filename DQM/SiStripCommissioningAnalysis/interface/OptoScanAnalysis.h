#ifndef DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H
#define DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;

/** 
   @class OptoScanAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Histogram-based analysis for opto bias/gain scan.
*/
class OptoScanAnalysis : public CommissioningAnalysis {
  
 public:
  
  OptoScanAnalysis() {;}
  virtual ~OptoScanAnalysis() {;}
  
  class TProfiles {
  public:
    TProfile* g0d0_; // Histo for digital "low", gain setting 0
    TProfile* g0d1_; // Histo for digital "high", gain setting 0
    TProfile* g1d0_; // Histo for digital "low", gain setting 1
    TProfile* g1d1_; // Histo for digital "high", gain setting 1
    TProfile* g2d0_; // Histo for digital "low", gain setting 2
    TProfile* g2d1_; // Histo for digital "high", gain setting 2
    TProfile* g3d0_; // Histo for digital "low", gain setting 3
    TProfile* g3d1_; // Histo for digital "high", gain setting 3
    TProfiles() : 
      g0d0_(0), g0d1_(0), 
      g1d0_(0), g1d1_(0), 
      g2d0_(0), g2d1_(0), 
      g3d0_(0), g3d1_(0) {;}
    ~TProfiles() {;}
    void print( std::stringstream& );
  };
  
  class Monitorables : public CommissioningAnalysis::Monitorables {
  public:
    uint16_t lldBias_; // LLD bias setting
    uint16_t lldGain_; // LLD gain setting
    float gain_;       // Measured gain
    float error_;      // Error on measured gain
    float base_;       // Level of "baseline" [adc]
    float peak_;       // Level of tick mark "peak" [adc]
    float height_;     // Tick mark height [ADC]
    Monitorables() : 
      lldBias_(sistrip::invalid_), lldGain_(sistrip::invalid_),
      gain_(sistrip::invalid_), error_(sistrip::invalid_), 
      base_(sistrip::invalid_), peak_(sistrip::invalid_), 
      height_(sistrip::invalid_) {;}
    virtual ~Monitorables() {;}
    void print( std::stringstream& );
  };

  /** New method. */
  static void analysis( const TProfiles&, Monitorables& ); 
  static void deprecated( const TProfiles&, Monitorables& ); 

 private: 
  
  /** Takes a vector containing one TH1F of median tick height
      measurements and one of median tick base measurements vs. LLD
      bias. Both histograms correspond to a fixed gain setting. A
      vector of floating points is filled with two values: the first
      being the optimum bias setting for the LLD channel, and the
      second the gain measurement. */
  static void analysis( const std::vector<const TProfile*>& histos, 
			std::vector<float>& monitorables );
  
};

#endif // DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H

