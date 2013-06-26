#ifndef DQMOffline_PFTau_PFCandidateMonitor_h
#define DQMOffline_PFTau_PFCandidateMonitor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>
class PFCandidateMonitor : public Benchmark {

 public:

  PFCandidateMonitor( float dRMax = 0.3,
		      bool matchCharge = true, 
		      Benchmark::Mode mode=Benchmark::DEFAULT); 

  virtual ~PFCandidateMonitor();
  
  /// set the parameters locally                                   
   void setParameters(float dRMax, bool matchCharge, Benchmark::Mode mode,
		     float ptmin, float ptmax, float etamin, float etamax,
		     float phimin, float phimax, bool refHistoFlag);

  /// set the parameters accessing them from ParameterSet   
  void setParameters( const edm::ParameterSet& parameterSet);
  
  /// set directory (to use in ROOT)
  void setDirectory(TDirectory* dir);

  /// book histograms
  void setup();
  
  /// book histograms
  void setup(const edm::ParameterSet & parameterSet);

  /// fill histograms with all particle
  template< class T, class C>
  void fill(const T& candidateCollection,
	    const C& matchedCandCollection, float& minVal, float& maxVal);


  void fillOne(const reco::Candidate& cand);

 protected:
  CandidateBenchmark      candBench_;
  MatchCandidateBenchmark matchCandBench_;

  TH1F*  pt_ref_;
  TH1F*  eta_ref_;
  TH1F*  phi_ref_;

  float dRMax_;
  bool  matchCharge_;
  bool  createReferenceHistos_;
  bool  histogramBooked_;

};

#include "DQMOffline/PFTau/interface/Matchers.h"
template< class T, class C>
void PFCandidateMonitor::fill(const T& candCollection,
			const C& matchedCandCollection, float& minVal, float& maxVal) {
  

  std::vector<int> matchIndices;
  PFB::match( candCollection, matchedCandCollection, matchIndices, 
	      matchCharge_, dRMax_ );

  for (unsigned int i = 0; i < (candCollection).size(); i++) {
     const reco::Candidate& cand = candCollection[i];

    if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) continue;
    
    int iMatch = matchIndices[i];
    assert(iMatch< static_cast<int>(matchedCandCollection.size()));

    if( iMatch!=-1 ) {
      const reco::Candidate& matchedCand = matchedCandCollection[ iMatch ];
      if(!isInRange(matchedCand.pt(),matchedCand.eta(),matchedCand.phi() ) ) continue;
      float ptRes = (cand.pt() - matchedCand.pt())/matchedCand.pt();
      
      if (ptRes > maxVal) maxVal = ptRes;
      if (ptRes < minVal) minVal = ptRes;
 
      candBench_.fillOne(cand);
      matchCandBench_.fillOne(cand, matchedCand);
      if (createReferenceHistos_) fillOne(matchedCand);
    }
  }
}
#endif 
