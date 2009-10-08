#ifndef RecoParticleFlow_Benchmark_PFCandidateBenchmark_h
#define RecoParticleFlow_Benchmark_PFCandidateBenchmark_h

#include "RecoParticleFlow/Benchmark/interface/CandidateBenchmark.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

/// To plot specific PFCandidate quantities
class PFCandidateBenchmark : public CandidateBenchmark {

 public:

  typedef reco::PFCandidateCollection Collection;

  PFCandidateBenchmark();
  virtual ~PFCandidateBenchmark();

  /// book histograms
  void setup();
  
  /// fill histograms with all particles
  void fill(const Collection& pfCandCollection );

 protected:
  
  /// fill histograms with a given particle
  void fill(const reco::PFCandidate& pfCand);
  
  TH1F*   particleId_; 
  TH1F*   ecalEnergy_; 
  TH1F*   hcalEnergy_; 
  TH1F*   mva_e_pi_; 
  TH1F*   elementsInBlocksSize_;

};

#endif 
