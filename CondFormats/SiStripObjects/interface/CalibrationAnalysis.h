#ifndef CondFormats_SiStripObjects_CalibrationAnalysis_H
#define CondFormats_SiStripObjects_CalibrationAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

/**
   @class CalibrationAnalysis
   @author C. Delaere
   @brief Analysis for calibration runs
*/

class CalibrationAnalysis : public CommissioningAnalysis {
  
 public:
  
  CalibrationAnalysis( const uint32_t& key, 
		       const bool& deconv, 
		       int calchan );
  
  CalibrationAnalysis( const bool& deconv, 
		       int calchan );
  
  virtual ~CalibrationAnalysis() {;}
  
  friend class CalibrationAlgorithm;

  // values per strip and per APV
  inline const VVFloat& amplitude() const { return amplitude_; }
  inline const VVFloat& tail() const { return tail_; }
  inline const VVFloat& riseTime() const { return riseTime_; }
  inline const VVFloat& timeConstant() const { return timeConstant_; }
  inline const VVFloat& smearing() const { return smearing_; }
  inline const VVFloat& chi2() const { return chi2_; }

  // mean values per APV
  inline const VFloat& amplitudeMean() const { return mean_amplitude_; }
  inline const VFloat& tailMean() const { return mean_tail_; }
  inline const VFloat& riseTimeMean() const { return mean_riseTime_; }
  inline const VFloat& timeConstantMean() const { return mean_timeConstant_; }
  inline const VFloat& smearingMean() const { return mean_smearing_; }
  inline const VFloat& chi2Mean() const { return mean_chi2_; }

  // spread, min and max
  inline const VFloat& amplitudeSpread() const { return spread_amplitude_; }
  inline const VFloat& amplitudeMin() const { return min_amplitude_; }
  inline const VFloat& amplitudeMax() const { return max_amplitude_; }

  inline const VFloat& tailSpread() const { return spread_tail_; }
  inline const VFloat& tailMin() const { return min_tail_; }
  inline const VFloat& tailMax() const { return max_tail_; }

  inline const VFloat& riseTimeSpread() const { return spread_riseTime_; }
  inline const VFloat& riseTimeMin() const { return min_riseTime_; }
  inline const VFloat& riseTimeMax() const { return max_riseTime_; }

  inline const VFloat& timeConstantSpread() const { return spread_timeConstant_; }
  inline const VFloat& timeConstantMin() const { return min_timeConstant_; }
  inline const VFloat& timeConstantMax() const { return max_timeConstant_; }

  inline const VFloat& smearingSpread() const { return spread_smearing_; }
  inline const VFloat& smearingMin() const { return min_smearing_; }
  inline const VFloat& smearingMax() const { return max_smearing_; }

  inline const VFloat& chi2Spread() const { return spread_chi2_; }
  inline const VFloat& chi2Min() const { return min_chi2_; }
  inline const VFloat& chi2Max() const { return max_chi2_; }
  
  inline bool deconvMode() const { return deconv_; }
  inline int calchan() const { return calchan_; }

  void print( std::stringstream&, uint32_t not_used = 0 );
  
  void reset();
  
 private:
  
  /** Parameters extracted from the fit of pulse shape */
  VVFloat amplitude_, tail_, riseTime_, timeConstant_, smearing_, chi2_;

  VFloat  mean_amplitude_,mean_tail_,mean_riseTime_,mean_timeConstant_,mean_smearing_,mean_chi2_;

  VFloat  min_amplitude_,min_tail_,min_riseTime_,min_timeConstant_,min_smearing_,min_chi2_;

  VFloat  max_amplitude_,max_tail_,max_riseTime_,max_timeConstant_,max_smearing_,max_chi2_;

  VFloat  spread_amplitude_,spread_tail_,spread_riseTime_,spread_timeConstant_,spread_smearing_,spread_chi2_;

  /** fit mode: deconv or not ? */
  bool deconv_;

  /** calchan value used in that dataset */
  int calchan_;

  /** internal mode: cal scan or standard run */
  bool isScan_;
  

};

#endif // CondFormats_SiStripObjects_CalibrationAnalysis_H

