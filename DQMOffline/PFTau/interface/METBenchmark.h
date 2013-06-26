#ifndef RecoParticleFlow_Benchmark_METBenchmark_h
#define RecoParticleFlow_Benchmark_METBenchmark_h

#include "DQMOffline/PFTau/interface/Benchmark.h"

#include "DataFormats/METReco/interface/METFwd.h"

/// To plot MET quantities
class METBenchmark : public Benchmark {

 public:

  METBenchmark(Mode mode) : Benchmark(mode) {}
  virtual ~METBenchmark();

  /// book histograms
  void setup();

  /// fill a collection
  template< class C>
  void fill( const C& candidates); 
  
  /// fill histograms with a given particle
  void fillOne( const reco::MET& candidate); 

 protected:
  
  TH1F*   pt_;
  TH1F*   pt2_;
  TH1F*   px_;
  TH1F*   py_;
  TH1F*   phi_;
  TH1F*   sumEt_;
  TH1F*   sumEt2_;  
  TH1F*   etOverSumEt_;
  TH2F*   mex_VS_sumEt_;

};

template< class C>
void METBenchmark::fill(const C& candCollection) {

  for (unsigned int i = 0; i < candCollection.size(); ++i) {
    const reco::MET& cand = candCollection[i];
    fillOne(cand);
  }
}

#endif 
