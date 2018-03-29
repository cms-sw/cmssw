#ifndef DQM_SiStripCommissioningAnalysis_CalibrationAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_CalibrationAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class CalibrationAnalysis;
class TH1;
class TF1;

/**
   @class CalibrationAlgorithm
   @author C. Delaere
   @brief Algorithm for calibration runs
*/

class CalibrationAlgorithm : public CommissioningAlgorithm {
  
 public:
  
  CalibrationAlgorithm( const edm::ParameterSet & pset, CalibrationAnalysis* const );
  
  ~CalibrationAlgorithm() override {;}
  
  inline const Histo& histo( int i ) const { return histo_[i]; }
  
 private:

  CalibrationAlgorithm() {;}
  
  void extract( const std::vector<TH1*>& ) override;

  void analyse() override;

  void correctDistribution( TH1* ) const;

  TF1* fitPulse( TH1*, 
		 float rangeLow = 0, 
		 float rangeHigh = -1 );
  
  float maximum( TH1* );
  
  float turnOn( TF1* );
  
 private:
  
  /** pulse shape*/
  Histo histo_[32];

  /** Fitter in deconvolution mode */
  TF1* deconv_fitter_;
  /** Fitter in peak mode */
  TF1* peak_fitter_;
  
  CalibrationAnalysis* cal_;
  
};

#endif // DQM_SiStripCommissioningAnalysis_CalibrationAlgorithm_H

