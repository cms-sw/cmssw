#ifndef DQM_SiStripCommissioningAnalysis_FedCablingAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_FedCablingAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class FedCablingAnalysis;
class TH1;

/** 
   @class FedCablingAlgorithm
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FedCablingAlgorithm : public CommissioningAlgorithm {
  
 public:

  // ---------- Con(de)structors and typedefs ----------

  FedCablingAlgorithm( const edm::ParameterSet & pset, FedCablingAnalysis* const );

  ~FedCablingAlgorithm() override {;}
  
  /** Pointer to FED id histogram. */
  inline const Histo& hFedId() const;
  
  /** Pointer to FED channel histogram. */
  inline const Histo& hFedCh() const;
  
 private:
  
  FedCablingAlgorithm() {;}

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& ) override;

  /** Performs histogram anaysis. */
  void analyse() override;
  
 private:

  /** Histo containing FED id */
  Histo hFedId_;

  /** Histo containing FED channel */
  Histo hFedCh_;

};

const FedCablingAlgorithm::Histo& FedCablingAlgorithm::hFedId() const { return hFedId_; }
const FedCablingAlgorithm::Histo& FedCablingAlgorithm::hFedCh() const { return hFedCh_; }

#endif // DQM_SiStripCommissioningAnalysis_FedCablingAlgorithm_H

