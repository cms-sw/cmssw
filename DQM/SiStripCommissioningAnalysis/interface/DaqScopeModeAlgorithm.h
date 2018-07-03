#ifndef DQM_SiStripCommissioningAnalysis_DaqScopeModeAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_DaqScopeModeAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class DaqScopeModeAnalysis;

/**
   @class DaqScopeModeAlgorithm
   @author R.Bainbridge
   @brief Algorithm for scope mode data.
*/

class DaqScopeModeAlgorithm : public CommissioningAlgorithm {
  
 public:
  
  DaqScopeModeAlgorithm( const edm::ParameterSet & pset, DaqScopeModeAnalysis* const );

  ~DaqScopeModeAlgorithm() override {;} 
  
  inline const Histo& hPeds() const;
  inline const Histo& hNoise() const;
  inline const Histo& histo() const;
  
 private:
  
  DaqScopeModeAlgorithm() {;}
  
  void extract( const std::vector<TH1*>& ) override;

  void analyse() override;
  
 private:
  
  /** Histogram of scope mode data. */
  Histo histo_;
  /** Pedestals and raw noise */
  Histo hPeds_;
  /** Residuals and noise */
  Histo hNoise_;
  
  /** Analysis parameters */
  float deadStripMax_;
  float noisyStripMin_;
  
};

const DaqScopeModeAlgorithm::Histo& DaqScopeModeAlgorithm::histo() const { return histo_; }
const DaqScopeModeAlgorithm::Histo& DaqScopeModeAlgorithm::hPeds() const { return hPeds_; }
const DaqScopeModeAlgorithm::Histo& DaqScopeModeAlgorithm::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_DaqScopeModeAlgorithm_H



