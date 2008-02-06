#ifndef CondFormats_SiStripObjects_CalibrationAnalysis_H
#define CondFormats_SiStripObjects_CalibrationAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <sstream>
#include <vector>

class TH1;
class TF1;

/**
   @class CalibrationAnalysis
   @author C. Delaere
   @brief Analysis for calibration runs
*/

class CalibrationAnalysis : public CommissioningAnalysis {
  
 public:
  
  CalibrationAnalysis( const uint32_t& key, const bool& deconv );
  CalibrationAnalysis(const bool& deconv = false);
  virtual ~CalibrationAnalysis() {;}
  
  inline const float& amplitude() const { return amplitude_; }
  inline const float& tail() const { return tail_; }
  inline const float& riseTime() const { return riseTime_; }
  inline const float& timeConstant() const { return timeConstant_; }
  inline const float& smearing() const { return smearing_; }
  inline const float& chi2() const { return chi2_; }
  
  inline const Histo& histo() const;
  void print( std::stringstream&, uint32_t not_used = 0 );
  
 private:
  
  void reset();
  virtual void extract( const std::vector<TH1*>& );
  void analyse();
  void correctDistribution(TH1* histo) const;
  TF1* fitPulse(TH1*);
  
 private:
  
  /** Parameters extracted from the fit of pulse shape */
  float amplitude_,tail_,riseTime_,timeConstant_,smearing_,chi2_;
  /** pulse shape*/
  Histo histo_;
  /** Fitter in deconvolution mode */
  TF1* deconv_fitter_;
  /** Fitter in peak mode */
  TF1* peak_fitter_;
  /** fit mode: deconv or not ? */
  bool deconv_;
  
};

#endif // CondFormats_SiStripObjects_CalibrationAnalysis_H

