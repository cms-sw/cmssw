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
  
  inline const float& entries() const;
  inline const float& mean() const; 
  inline const float& median() const; 
  inline const float& mode() const; 
  inline const float& rms() const; 
  inline const float& min() const; 
  inline const float& max() const; 
  
  inline const Histo& histo() const;
  
 private:
  
  DaqScopeModeAlgorithm() {;}
  
  void extract( const std::vector<TH1*>& ) override;

  void analyse() override;
  
 private:
  
  /** Histogram of scope mode data. */
  Histo histo_;
  
};
const DaqScopeModeAlgorithm::Histo& DaqScopeModeAlgorithm::histo() const { return histo_; }

#endif // DQM_SiStripCommissioningAnalysis_DaqScopeModeAlgorithm_H



