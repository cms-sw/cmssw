#ifndef DQM_SiStripCommissioningAnalysis_PedsFullNoiseAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_PedsFullNoiseAlgorithm_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAlgorithm.h"
#include <vector>

class PedsFullNoiseAnalysis;
class TH1;

/** 
    @class PedsFullNoiseAlgorithm
    @author M. Wingham, R.Bainbridge
    @brief Histogram-based analysis for pedestal run.
*/
class PedsFullNoiseAlgorithm : public CommissioningAlgorithm {
  
 public:

  PedsFullNoiseAlgorithm( const edm::ParameterSet & pset, PedsFullNoiseAnalysis* const );

  virtual ~PedsFullNoiseAlgorithm() {;}

  inline const Histo& hPeds() const;

  inline const Histo& hNoise() const;
  
  inline const Histo& hNoise1D() const;

 private:

  PedsFullNoiseAlgorithm() {;}

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& );

  /** Performs histogram anaysis. */
  void analyse();

 private:

  /** Pedestals and raw noise */
  Histo hPeds_;

  /** Residuals and noise */
  Histo hNoise_;
  Histo hNoise1D_;
  
  /** Analysis parameters */
  float deadStripMax_;
  float noisyStripMin_;
  std::string noiseDef_;
  float ksProbCut_;
};

const PedsFullNoiseAlgorithm::Histo& PedsFullNoiseAlgorithm::hPeds() const { return hPeds_; }

const PedsFullNoiseAlgorithm::Histo& PedsFullNoiseAlgorithm::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_PedsFullNoiseAlgorithm_H
