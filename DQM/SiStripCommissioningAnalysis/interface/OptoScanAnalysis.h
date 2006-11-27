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
  
  OptoScanAnalysis( const uint32_t& key );
  OptoScanAnalysis();
  virtual ~OptoScanAnalysis() {;}
  
  inline const uint16_t& gain() const;
  inline const VInts& bias() const;
  inline const VFloats& measGain() const;
  inline const VFloats& zeroLight() const;
  inline const VFloats& linkNoise() const;
  inline const VFloats& liftOff() const;
  inline const VFloats& threshold() const;
  inline const VFloats& tickHeight() const;
  
  void print( std::stringstream&, uint32_t gain_setting = 0 );

 private:
  
  void reset();
  void extract( const std::vector<TH1*>& );
  void analyse();

 private:

  /** Optimum LLD gain setting */
  uint16_t gain_;
  /** LLD bias for each gain setting */
  VInts bias_;
  /** Measured gains [adc] */
  VFloats measGain_;
  /** "Zero light" level [adc] */
  VFloats zeroLight_;
  /** Noise at "zero light" level [adc] */
  VFloats linkNoise_;
  /** Baseline "lift-off" [mA] */
  VFloats liftOff_;
  /** Laser threshold [mA] */
  VFloats threshold_;
  /** Tick mark height [adc] */
  VFloats tickHeight_;

  /** Histo for digital "low", gain setting 0 */
  Histo g0d0_;
  /** Histo for digital "high", gain setting 0 */
  Histo g0d1_;
  /** Histo for digital "low", gain setting 1 */
  Histo g1d0_;
  /** Histo for digital "high", gain setting 1 */
  Histo g1d1_;
  /** Histo for digital "low", gain setting 2 */
  Histo g2d0_;
  /** Histo for digital "high", gain setting 2 */
  Histo g2d1_;
  /** Histo for digital "low", gain setting 3 */
  Histo g3d0_;
  /** Histo for digital "high", gain setting 3 */
  Histo g3d1_;
  
 private:

  void deprecated(); 
  void anal( const std::vector<const TProfile*>& histos, 
	     std::vector<float>& monitorables );
  
};
 
const uint16_t& OptoScanAnalysis::gain() const { return gain_; }
const OptoScanAnalysis::VInts& OptoScanAnalysis::bias() const { return bias_; }
const OptoScanAnalysis::VFloats& OptoScanAnalysis::measGain() const { return measGain_; }
const OptoScanAnalysis::VFloats& OptoScanAnalysis::zeroLight() const { return zeroLight_; }
const OptoScanAnalysis::VFloats& OptoScanAnalysis::linkNoise() const { return linkNoise_; }
const OptoScanAnalysis::VFloats& OptoScanAnalysis::liftOff() const { return liftOff_; }
const OptoScanAnalysis::VFloats& OptoScanAnalysis::threshold() const { return threshold_; }
const OptoScanAnalysis::VFloats& OptoScanAnalysis::tickHeight() const { return tickHeight_; }

#endif // DQM_SiStripCommissioningAnalysis_OptoScanAnalysis_H

