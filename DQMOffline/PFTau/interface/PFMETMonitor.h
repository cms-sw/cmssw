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

  PFMETMonitor( Benchmark::Mode mode=Benchmark::DEFAULT); 
  
  virtual ~PFMETMonitor();

  /// set the parameters locally
  void setParameters(Benchmark::Mode mode, float ptmin, float ptmax, float etamin,
                    float etamax, float phimin, float phimax, bool metSpHistos);
  
  /// set the parameters accessing them from ParameterSet
  void setParameters( const edm::ParameterSet& parameterSet);
  
  /// set directory (to use in ROOT)
  void setDirectory(TDirectory* dir);

  /// book histograms
  void setup(DQMStore::IBooker& b);
  void setup(DQMStore::IBooker& b, const edm::ParameterSet & parameterSet);

  void fillOne(const reco::MET& met, const reco::MET& matchedMet,
	       float& minVal, float& maxVal);

  void fillOne(const reco::MET& met, const reco::MET& matchedMet,
	       float& minVal, float& maxVal,
	       const edm::ParameterSet & parameterSet);

 protected:
  TH1F* px_;
  TH1F* sumEt_;
  TH1F* delta_ex_;
  TH2F* delta_ex_VS_set_;
  TH2F* delta_set_VS_set_;
  TH2F* delta_set_Over_set_VS_set_;

  TProfile* profile_delta_ex_VS_set_;
  TProfile* profile_delta_set_VS_set_;
  TProfile* profile_delta_set_Over_set_VS_set_;

  TProfile* profileRMS_delta_ex_VS_set_;
  TProfile* profileRMS_delta_set_VS_set_;
  TProfile* profileRMS_delta_set_Over_set_VS_set_;


  CandidateBenchmark      candBench_;
  MatchCandidateBenchmark matchCandBench_;

  bool  createMETSpecificHistos_;
  bool  histogramBooked_;
};
#endif 
