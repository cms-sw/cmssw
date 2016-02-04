#ifndef DQMOffline_PFTau_PFMETMonitor_h
#define DQMOffline_PFTau_PFMETMonitor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include <vector>
class PFMETMonitor : public Benchmark {

 public:

  PFMETMonitor( Benchmark::Mode mode=Benchmark::DEFAULT) 
    : 
    Benchmark(mode), 
    candBench_(mode), matchCandBench_(mode) {} 
  
  virtual ~PFMETMonitor();
  
  /// set the parameters
  void setParameters( const edm::ParameterSet& parameterSet);
  
  /// set directory (to use in ROOT)
  void setDirectory(TDirectory* dir);

  /// book histograms
  void setup();
  
  /// book histograms
  void setup(const edm::ParameterSet & parameterSet);

  void fillOne(const reco::MET& met,
	       const reco::MET& matchedMet, float& minVal, float& maxVal);

 protected:
  TH1F*   px_;
  TH1F*   sumEt_;
  TH1F*   delta_ex_;
  TH2F*   delta_ex_VS_set_;
  TH2F*   delta_set_VS_set_;
  TH2F*   delta_set_Over_set_VS_set_;

  std::vector<double> variablePtBins_;
 
  CandidateBenchmark      candBench_;
  MatchCandidateBenchmark matchCandBench_;

  bool  createMETSpecificHistos_;
  bool  histogramBooked_;
};
#endif 
