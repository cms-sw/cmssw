#ifndef DQMOffline_PFTau_PFJetMonitor_h
#define DQMOffline_PFTau_PFJetMonitor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/JetReco/interface/BasicJetCollection.h"

#include <vector>

#include <TH1.h>  //needed by the deltaR->Fill() call

class PFJetMonitor : public Benchmark {

 public:

  
  PFJetMonitor( float dRMax = 0.3, bool matchCharge = true, 
		Benchmark::Mode mode=Benchmark::DEFAULT); 
  
  virtual ~PFJetMonitor();
  
  /// set the parameters locally
  void setParameters(float dRMax, bool matchCharge, Benchmark::Mode mode,
		     float ptmin, float ptmax, float etamin, float etamax, float phimin, float phimax,
		     bool fracHistoFlag=true);
  
  void setParameters(float dRMax, bool onlyTwoJets, bool matchCharge, Benchmark::Mode mode,
		     float ptmin, float ptmax, float etamin, float etamax, float phimin, float phimax,
		     bool fracHistoFlag=true);
  
  /// set the parameters accessing them from ParameterSet
  void setParameters( const edm::ParameterSet& parameterSet);
  
  /// set directory (to use in ROOT)
  void setDirectory(TDirectory* dir);

  /// book histograms
  void setup(DQMStore::IBooker& b);
  void setup(DQMStore::IBooker& b, const edm::ParameterSet & parameterSet);
  
  /// fill histograms with all particle
  template< class T, class C>
    void fill(const T& jetCollection, const C& matchedJetCollection,
	      float& minVal, float& maxVal);
  
  template< class T, class C>
    void fill(const T& candidateCollection, const C& matchedCandCollection,
	      float& minVal, float& maxVal, float& jetpT,
	      const edm::ParameterSet & parameterSet);

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

  TH1F* deltaR_;
  float dRMax_;
  bool  onlyTwoJets_;
  bool  matchCharge_;
  bool  createPFractionHistos_;
  bool  histogramBooked_;

};

#include "DQMOffline/PFTau/interface/Matchers.h"
template< class T, class C>
  void PFJetMonitor::fill(const T& jetCollection,
			const C& matchedJetCollection, float& minVal, float& maxVal) {
  
  
  std::vector<int> matchIndices;
  PFB::match( jetCollection, matchedJetCollection, matchIndices, matchCharge_, dRMax_ );
  
  for (unsigned int i = 0; i < (jetCollection).size(); i++) {
    const reco::Jet& jet = jetCollection[i];
    
    if( !isInRange(jet.pt(), jet.eta(), jet.phi() ) ) continue;
    
    int iMatch = matchIndices[i];
    assert(iMatch< static_cast<int>(matchedJetCollection.size()));
    
    if( iMatch != -1 ) {
      const reco::Candidate& matchedJet = matchedJetCollection[ iMatch ];
      if( !isInRange( matchedJet.pt(), matchedJet.eta(), matchedJet.phi() ) ) continue;
      float ptRes = (jet.pt() - matchedJet.pt())/matchedJet.pt();
      
      if (ptRes > maxVal) maxVal = ptRes;
      if (ptRes < minVal) minVal = ptRes;
      
      candBench_.fillOne(jet);
      matchCandBench_.fillOne(jet, matchedJetCollection[ iMatch ]);
      if (createPFractionHistos_ && histogramBooked_) fillOne(jet, matchedJetCollection[ iMatch ]);
    }
  }
}


template< class T, class C>
  void PFJetMonitor::fill(const T& jetCollection,
			  const C& matchedJetCollection, float& minVal, float& maxVal, float& jetpT,
			  const edm::ParameterSet & parameterSet) {
  
  std::vector<int> matchIndices;
  PFB::match( jetCollection, matchedJetCollection, matchIndices, matchCharge_, dRMax_ );
  // now matchIndices[i] stores the j-th closest matched jet 

  for( unsigned i=0; i<jetCollection.size(); ++i) {
    // Count the number of jets with a larger energy = pT
    unsigned int highJets = 0;
    for( unsigned j=0; j<jetCollection.size(); ++j) {
      if (j != i && jetCollection[j].pt() > jetCollection[i].pt()) highJets++;
    }    
    if ( onlyTwoJets_ && highJets > 1 ) continue;

    const reco::Jet& jet = jetCollection[i];
    
    if( !isInRange(jet.pt(), jet.eta(), jet.phi() ) ) continue;
    
    int iMatch = matchIndices[i];
    assert( iMatch < static_cast<int>(matchedJetCollection.size()) );
    
    if( iMatch != -1 ) {
      const reco::Candidate& matchedJet = matchedJetCollection[ iMatch ];
      if ( !isInRange( matchedJet.pt(), matchedJet.eta(), matchedJet.phi() ) ) continue;
      
      float ptRes = (jet.pt() - matchedJet.pt()) / matchedJet.pt();
      
      jetpT = jet.pt();
      if (ptRes > maxVal) maxVal = ptRes;
      if (ptRes < minVal) minVal = ptRes;
      
      candBench_.fillOne(jet);  // fill pt eta phi and charge histos for MATCHED candidate jet
      matchCandBench_.fillOne(jet, matchedJetCollection[iMatch], parameterSet);  // fill delta_x_VS_y histos for matched couple
      if (createPFractionHistos_ && histogramBooked_) fillOne(jet, matchedJetCollection[iMatch]);  // book and fill delta_frac_VS_frac histos for matched couple
    }
        
    for( unsigned j=0; j<matchedJetCollection.size(); ++j)  // for DeltaR spectrum
      if (deltaR_) deltaR_->Fill( reco::deltaR( jetCollection[i], matchedJetCollection[j] ) ) ;
  } // end loop on jetCollection
}
#endif 
