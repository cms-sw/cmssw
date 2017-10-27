#ifndef DQM_SiStripCommissioningAnalysis_FedTimingAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_FedTimingAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class FedTimingAnalysis;
class TH1;

/**
   @class FedTimingAlgorithm
   @author M. Wingham, R.Bainbridge
   @brief Algorithm for timing run using APV tick marks.
*/

class FedTimingAlgorithm : public CommissioningAlgorithm {
  
 public:
  
  FedTimingAlgorithm( const edm::ParameterSet & pset, FedTimingAnalysis* const );

  ~FedTimingAlgorithm() override {;}
  
  inline const Histo& histo() const;
  
 private:

  FedTimingAlgorithm() {;}
  
  void extract( const std::vector<TH1*>& ) override;

  void analyse() override;
  
 private:
  
  /** APV tick mark */
  Histo histo_;
  
};

const FedTimingAlgorithm::Histo& FedTimingAlgorithm::histo() const { return histo_; }

#endif // DQM_SiStripCommissioningAnalysis_FedTimingAlgorithm_H



