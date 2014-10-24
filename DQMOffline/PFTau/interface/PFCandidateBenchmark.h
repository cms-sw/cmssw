#ifndef RecoParticleFlow_Benchmark_PFCandidateBenchmark_h
#define RecoParticleFlow_Benchmark_PFCandidateBenchmark_h

#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

/// To plot specific PFCandidate quantities
/// the name of the histograms corresponds to the name of the
/// PFCandidate accessors. 
class PFCandidateBenchmark : public Benchmark {

 public:

  PFCandidateBenchmark(Mode mode) : Benchmark(mode) {}
  virtual ~PFCandidateBenchmark();

  /// book histograms
  void setup();
  
  void fill( const reco::PFCandidateCollection& pfCands);

  /// fill histograms with a given particle
  void fillOne(const reco::PFCandidate& pfCand);

 protected:
  
  TH1F*   particleId_; 
  TH1F*   ecalEnergy_; 
  TH1F*   hcalEnergy_; 
  TH1F*   mva_e_pi_; 
  TH1F*   elementsInBlocksSize_;

};

#endif 
