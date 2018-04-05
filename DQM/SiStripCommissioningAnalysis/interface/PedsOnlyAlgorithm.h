#ifndef DQM_SiStripCommissioningAnalysis_PedsOnlyAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_PedsOnlyAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class PedsOnlyAnalysis;
class TH1;

/** 
    @class PedsOnlyAlgorithm
    @author M. Wingham, R.Bainbridge
    @brief Histogram-based analysis for pedestal run.
*/
class PedsOnlyAlgorithm : public CommissioningAlgorithm {
  
 public:

  PedsOnlyAlgorithm( const edm::ParameterSet & pset, PedsOnlyAnalysis* const );

  ~PedsOnlyAlgorithm() override {;}

  inline const Histo& hPeds() const;

  inline const Histo& hNoise() const;

 private:
  
  PedsOnlyAlgorithm() {;}

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& ) override;

  /** Performs histogram anaysis. */
  void analyse() override;
  
 private:

  /** Pedestals and raw noise */
  Histo hPeds_;

  /** Residuals and noise */
  Histo hNoise_;
  
};

const PedsOnlyAlgorithm::Histo& PedsOnlyAlgorithm::hPeds() const { return hPeds_; }

const PedsOnlyAlgorithm::Histo& PedsOnlyAlgorithm::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_PedsOnlyAlgorithm_H
