#ifndef DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H
#define DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H

#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TProfile;
class TH1;

/** 
   @class OptoScanAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Histogram-based analysis for opto bias/gain scan.
*/
class OptoScanAnalysis : public CommissioningAnalysis {
  
 public:
  
  // ---------- Con(de)structors ----------

  OptoScanAnalysis( const uint32_t& key );
  OptoScanAnalysis();
  virtual ~OptoScanAnalysis() {;}

  // ---------- Analysis results and histos ----------
  
  /** Optimum LLD gain setting */
  inline const uint16_t& gain() const;
  
  /** LLD bias value for each gain setting */
  inline const VInt& bias() const;
  
  /** Measured gains for each setting [V/V]. */
  inline const VFloat& measGain() const;
  
  /** "Zero light" levels [ADC] */
  inline const VFloat& zeroLight() const;

  /** Noise value at "zero light" levels [ADC] */
  inline const VFloat& linkNoise() const;

  /** Baseline "lift-off" values [mA] */
  inline const VFloat& liftOff() const;

  /** Laser thresholds [mA] */
  inline const VFloat& threshold() const;

  /** Tick mark heights [ADC] */
  inline const VFloat& tickHeight() const;

  /** Histogram pointer and title. */
  Histo histo( const uint16_t& gain, 
	       const uint16_t& digital_level ) const;
  
  // ---------- Utility methods ----------
  
  /** Identifies if analysis is valid or not. */
  bool isValid();

  /** Prints analysis results. */
  void print( std::stringstream&, uint32_t gain_setting = 0 );
  
  // ---------- Private methods ----------
  
  /** Resets analysis member data. */
  void reset();

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& );

  /** Performs histogram anaysis. */
  void analyse();
  
 private:

  // ---------- Private member data ----------

  /** Optimum LLD gain setting */
  uint16_t gain_;

  /** LLD bias value for each gain setting */
  VInt bias_;

  /** Measured gains for each setting [V/V]. */
  VFloat measGain_;
  
  /** "Zero light" levels [ADC] */
  VFloat zeroLight_;

  /** Noise value at "zero light" levels [ADC] */
  VFloat linkNoise_;

  /** Baseline "lift-off" values [mA] */
  VFloat liftOff_;

  /** Laser thresholds [mA] */
  VFloat threshold_;

  /** Tick mark heights [ADC] */
  VFloat tickHeight_;

  /** Default LLD gain setting if analysis fails. */
  static const uint16_t defaultGainSetting_;

  /** Default LLD bias setting if analysis fails. */
  static const uint16_t defaultBiasSetting_;

  /** Peak-to-peak voltage for FED A/D converter [V/ADC]. */
  static const float fedAdcGain_;
  
  /** Pointers and titles for histograms. */
  std::vector< std::vector<Histo> > opto_;
  
 private:

  // ---------- Private deprecated or wrapper methods ----------
  
  void deprecated(); 
  
  void anal( const std::vector<const TProfile*>& histos, 
	     std::vector<float>& monitorables );
  
};

// ---------- Inline methods ----------
 
const uint16_t& OptoScanAnalysis::gain() const { return gain_; }
const OptoScanAnalysis::VInt& OptoScanAnalysis::bias() const { return bias_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::measGain() const { return measGain_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::zeroLight() const { return zeroLight_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::linkNoise() const { return linkNoise_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::liftOff() const { return liftOff_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::threshold() const { return threshold_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::tickHeight() const { return tickHeight_; }

#endif // DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H

