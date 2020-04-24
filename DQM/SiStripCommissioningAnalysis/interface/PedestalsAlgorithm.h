#ifndef DQM_SiStripCommissioningAnalysis_PedestalsAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_PedestalsAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class PedestalsAnalysis;
class TH1;

/** 
    @class PedestalsAlgorithm
    @author M. Wingham, R.Bainbridge
    @brief Histogram-based analysis for pedestal run.
*/
class PedestalsAlgorithm : public CommissioningAlgorithm {
  
 public:

  PedestalsAlgorithm( const edm::ParameterSet & pset, PedestalsAnalysis* const );

  ~PedestalsAlgorithm() override {;}

  inline const Histo& hPeds() const;

  inline const Histo& hNoise() const;

 private:

  PedestalsAlgorithm() {;}

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& ) override;

  /** Performs histogram anaysis. */
  void analyse() override;

 private:

  /** Pedestals and raw noise */
  Histo hPeds_;

  /** Residuals and noise */
  Histo hNoise_;
  
  /** Analysis parameters */
  float deadStripMax_;
  float noisyStripMin_;
  
};

const PedestalsAlgorithm::Histo& PedestalsAlgorithm::hPeds() const { return hPeds_; }

const PedestalsAlgorithm::Histo& PedestalsAlgorithm::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_PedestalsAlgorithm_H
