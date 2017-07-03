#ifndef DQM_SiStripCommissioningAnalysis_FastFedCablingAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_FastFedCablingAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class FastFedCablingAnalysis;
class TH1;

/** 
   @class FastFedCablingAlgorithm
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FastFedCablingAlgorithm : public CommissioningAlgorithm {
  
 public:

  FastFedCablingAlgorithm( const edm::ParameterSet & pset, FastFedCablingAnalysis* const );
  
  ~FastFedCablingAlgorithm() override {;}
  
  /** Container of histogram pointer and title. */
  inline const Histo& histo() const;
  
 private:
  
  /** Private constructor. */
  FastFedCablingAlgorithm() {;}
  
  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& ) override;

  /** Performs histogram anaysis. */
  void analyse() override;

 private:
  
  /** Histo */
  Histo histo_;

};

// ---------- Inline methods ----------
  
const FastFedCablingAlgorithm::Histo& FastFedCablingAlgorithm::histo() const { return histo_; }

#endif // DQM_SiStripCommissioningAnalysis_FastFedCablingAlgorithm_H


