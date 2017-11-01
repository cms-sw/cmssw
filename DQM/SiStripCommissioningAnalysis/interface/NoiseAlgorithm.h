#ifndef DQM_SiStripCommissioningAnalysis_NoiseAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_NoiseAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class NoiseAnalysis;
class TH1;

/** 
    @class NoiseAlgorithm
    @author M. Wingham, R.Bainbridge
    @brief Histogram-based analysis for pedestal run.
*/
class NoiseAlgorithm : public CommissioningAlgorithm {
  
 public:

  NoiseAlgorithm( const edm::ParameterSet & pset, NoiseAnalysis* const );

  ~NoiseAlgorithm() override {;}

  inline const Histo& hPeds() const;

  inline const Histo& hNoise() const;

 private:

  NoiseAlgorithm() {;}

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& ) override;

  /** Performs histogram anaysis. */
  void analyse() override;
  
  // ---------- private member data ----------

 private:

  /** Pedestals and raw noise */
  Histo hPeds_;

  /** Residuals and noise */
  Histo hNoise_;
  
};

const NoiseAlgorithm::Histo& NoiseAlgorithm::hPeds() const { return hPeds_; }

const NoiseAlgorithm::Histo& NoiseAlgorithm::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_NoiseAlgorithm_H
