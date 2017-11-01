#ifndef RecoParticleFlow_Benchmark_MatchCandidateBenchmark_h
#define RecoParticleFlow_Benchmark_MatchCandidateBenchmark_h

#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

/// To plot Candidate quantities
class MatchCandidateBenchmark : public Benchmark {

 public:

  MatchCandidateBenchmark(Mode mode);

  ~MatchCandidateBenchmark() override;

  /// book histograms
  void setup(DQMStore::IBooker& b);
  void setup(DQMStore::IBooker& b, const edm::ParameterSet& parameterSet);
  
  /// fill histograms with a given particle
  void fillOne( const reco::Candidate& candidate,
    const reco::Candidate& matchedCandidate ); 
  
  void fillOne( const reco::Candidate& candidate,
		const reco::Candidate& matchedCandidate,
		const edm::ParameterSet& parameterSet );
  
 protected:
  
  TH2F* delta_et_Over_et_VS_et_;
  TH2F* delta_et_VS_et_; 
  TH2F* delta_eta_VS_et_; 
  TH2F* delta_phi_VS_et_;

  TH2F* BRdelta_et_Over_et_VS_et_; 
  TH2F* ERdelta_et_Over_et_VS_et_;
  std::vector<TH1F*> pTRes_ ;
  std::vector<TH1F*> BRpTRes_ ;
  std::vector<TH1F*> ERpTRes_ ;

  bool  histogramBooked_;

};

 

#endif 
