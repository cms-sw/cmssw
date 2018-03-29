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

  ~PedsFullNoiseAlgorithm() override {;}

  inline const Histo& hPeds() const;
  inline const Histo& hNoise() const;
  inline const Histo& hNoise2D() const;
  

 private:

  PedsFullNoiseAlgorithm() {;}

  /** Extracts and organises histograms. */
  void extract( const std::vector<TH1*>& ) override;

  /** Performs histogram anaysis. */
  void analyse() override;

  /** reset vector */
  void reset(PedsFullNoiseAnalysis*);     

 private:

  /** Pedestals and raw noise */
  Histo hPeds_;
  /** Noise and residuals */
  Histo hNoise_;
  Histo hNoise2D_;
  
  /** Analysis parameters */
  float maxDriftResidualCut_;
  float minStripNoiseCut_;
  float maxStripNoiseCut_;
  float maxStripNoiseSignificanceCut_;
  float adProbabCut_;
  float ksProbabCut_;
  bool  generateRandomHisto_;
  float jbProbabCut_;
  float chi2ProbabCut_;
  float kurtosisCut_;
  float integralTailCut_;
  int   integralNsigma_;
  float ashmanDistance_;
  float amplitudeRatio_;

};

const PedsFullNoiseAlgorithm::Histo& PedsFullNoiseAlgorithm::hPeds() const { return hPeds_; }
const PedsFullNoiseAlgorithm::Histo& PedsFullNoiseAlgorithm::hNoise2D() const { return hNoise2D_; }
const PedsFullNoiseAlgorithm::Histo& PedsFullNoiseAlgorithm::hNoise() const { return hNoise_; }

#endif // DQM_SiStripCommissioningAnalysis_PedsFullNoiseAlgorithm_H
