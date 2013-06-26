#ifndef DQMOffline_PFTau_PFJetMonitor_h
#define DQMOffline_PFTau_PFJetMonitor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/JetReco/interface/BasicJetCollection.h"

#include <vector>
class PFJetMonitor : public Benchmark {

 public:

  
  PFJetMonitor( float dRMax = 0.3, bool matchCharge = true, 
		Benchmark::Mode mode=Benchmark::DEFAULT); 
  
  virtual ~PFJetMonitor();
  
  /// set the parameters locally
  void setParameters(float dRMax, bool matchCharge, 
                     Benchmark::Mode mode,float ptmin, 
                     float ptmax, float etamin, float etamax, 
                     float phimin, float phimax, bool fracHistoFlag=true);

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
  void fill(const T& jetCollection,
	    const C& matchedJetCollection, float& minVal, float& maxVal);

  void fillOne(const reco::Jet& jet,
	       const reco::Jet& matchedJet);

 protected:
  CandidateBenchmark      candBench_;
  MatchCandidateBenchmark matchCandBench_;

  TH2F*  delta_frac_VS_frac_muon_;
  TH2F*  delta_frac_VS_frac_photon_;
  TH2F*  delta_frac_VS_frac_electron_;
  TH2F*  delta_frac_VS_frac_charged_hadron_;
  TH2F*  delta_frac_VS_frac_neutral_hadron_;

  float dRMax_;
  bool  matchCharge_;
  bool  createPFractionHistos_;
  bool  histogramBooked_;

};

#include "DQMOffline/PFTau/interface/Matchers.h"
template< class T, class C>
void PFJetMonitor::fill(const T& jetCollection,
			const C& matchedJetCollection, float& minVal, float& maxVal) {
  

  std::vector<int> matchIndices;
  PFB::match( jetCollection, matchedJetCollection, matchIndices, 
	      matchCharge_, dRMax_ );

  for (unsigned int i = 0; i < (jetCollection).size(); i++) {
    const reco::Jet& jet = jetCollection[i];

    if( !isInRange(jet.pt(), jet.eta(), jet.phi() ) ) continue;
    
    int iMatch = matchIndices[i];
    assert(iMatch< static_cast<int>(matchedJetCollection.size()));

    if( iMatch!=-1 ) {
      const reco::Candidate& matchedJet = matchedJetCollection[ iMatch ];
      if(!isInRange(matchedJet.pt(),matchedJet.eta(),matchedJet.phi() ) ) continue;
      float ptRes = (jet.pt() - matchedJet.pt())/matchedJet.pt();
      
      if (ptRes > maxVal) maxVal = ptRes;
      if (ptRes < minVal) minVal = ptRes;
 
      candBench_.fillOne(jet);
      matchCandBench_.fillOne(jet, matchedJetCollection[ iMatch ]);
      if (createPFractionHistos_ && histogramBooked_) fillOne(jet, matchedJetCollection[ iMatch ]);
    }
  }
}
#endif 
