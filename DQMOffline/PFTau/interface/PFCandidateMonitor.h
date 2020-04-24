#ifndef DQMOffline_PFTau_PFCandidateMonitor_h
#define DQMOffline_PFTau_PFCandidateMonitor_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMOffline/PFTau/interface/Benchmark.h"
#include "DQMOffline/PFTau/interface/CandidateBenchmark.h"
#include "DQMOffline/PFTau/interface/MatchCandidateBenchmark.h"

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include <vector>

#include <TH1.h>  //needed by the deltaR->Fill() call

class PFCandidateMonitor : public Benchmark {

 public:

  PFCandidateMonitor( float dRMax = 0.3, bool matchCharge = true, 
		      Benchmark::Mode mode=Benchmark::DEFAULT); 

  ~PFCandidateMonitor() override;
  
  /// set the parameters locally                                   
  void setParameters(float dRMax, bool matchCharge, Benchmark::Mode mode,
		     float ptmin, float ptmax, float etamin, float etamax,
		     float phimin, float phimax, bool refHistoFlag);

  /// set the parameters accessing them from ParameterSet   
  void setParameters( const edm::ParameterSet& parameterSet);
  
  /// set directory (to use in ROOT)
  void setDirectory(TDirectory* dir) override;

  /// book histograms
  void setup(DQMStore::IBooker& b);
  void setup(DQMStore::IBooker& b, const edm::ParameterSet & parameterSet);
  
  /// fill histograms with all particle
  template< class T, class C>  
    /*void fill(const T& candidateCollection,
      const C& matchedCandCollection, float& minVal, float& maxVal) ;*/
    void fill(const T& candidateCollection,
	      const C& matchedCandCollection, float& minVal, float& maxVal,
	      const edm::ParameterSet & parameterSet) ;
  template< class T, class C, class M>  
    void fill(const T& candidateCollection,
	      const C& matchedCandCollection, float& minVal, float& maxVal,
	      const edm::ParameterSet & parameterSet, const M& muonMatchedCandCollection ) ;
  
  void fillOne(const reco::Candidate& cand);
  
 protected:
  CandidateBenchmark      candBench_;
  MatchCandidateBenchmark matchCandBench_;

  TH1F* pt_gen_;
  TH1F* eta_gen_;
  TH1F* phi_gen_;

  TH1F* pt_ref_;
  TH1F* eta_ref_;
  TH1F* phi_ref_;

  TH1F* deltaR_;
  float dRMax_;
  bool  matchCharge_;
  bool  createReferenceHistos_;
  bool  histogramBooked_;

  bool  matching_done_;
  bool  createEfficiencyHistos_;
};

#include "DQMOffline/PFTau/interface/Matchers.h"
template< class T, class C>
  void PFCandidateMonitor::fill(const T& candCollection,
				const C& matchedCandCollection, float& minVal, float& maxVal,
				const edm::ParameterSet & parameterSet) {
  
  matching_done_ = false;
  if ( createEfficiencyHistos_ ) {
    for( unsigned i=0; i<candCollection.size(); ++i) {
      if( !isInRange(candCollection[i].pt(), candCollection[i].eta(), candCollection[i].phi() ) ) continue;
      fillOne(candCollection[i]); // fill pt_gen, eta_gen and phi_gen histos for UNMATCHED generated candidate

      for( unsigned j=0; j<matchedCandCollection.size(); ++j)  // for DeltaR spectrum
	if (deltaR_) deltaR_->Fill( reco::deltaR( candCollection[i], matchedCandCollection[j] ) ) ;
    }
  }

  std::vector<int> matchIndices;
  PFB::match( candCollection, matchedCandCollection, matchIndices, matchCharge_, dRMax_ );
  //PFB::match( candCollection, matchedCandCollection, matchIndices, parameterSet, matchCharge_, dRMax_ );
  // now matchIndices[i] stores the j-th closest matched jet 
  matching_done_ = true;

  for (unsigned int i = 0; i < (candCollection).size(); i++) {
     const reco::Candidate& cand = candCollection[i];

    if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) continue;
    
    int iMatch = matchIndices[i];
    assert(iMatch< static_cast<int>(matchedCandCollection.size()));

    if( iMatch!=-1 ) {
      const reco::Candidate& matchedCand = matchedCandCollection[ iMatch ];
      if(!isInRange(matchedCand.pt(),matchedCand.eta(),matchedCand.phi() ) ) continue;
      //std::cout <<"PFJet pT " <<cand.pt() <<" eta " <<cand.eta() <<" phi " <<cand.phi() ;
      //std::cout <<"\nmatched genJet pT " <<matchedCand.pt() <<" eta " <<matchedCand.eta() <<" phi " <<matchedCand.phi() <<"\n" <<std::endl ;
      float ptRes = (cand.pt() - matchedCand.pt()) / matchedCand.pt();
      
      if (ptRes > maxVal) maxVal = ptRes;
      if (ptRes < minVal) minVal = ptRes;
 
      if ( !createEfficiencyHistos_ ) {
	candBench_.fillOne(cand);   // fill pt, eta phi and charge histos for MATCHED candidate
	//matchCandBench_.fillOne(cand, matchedCand);  // fill delta_x_VS_y histos for matched couple
	matchCandBench_.fillOne(cand, matchedCand, parameterSet);  // fill delta_x_VS_y histos for matched couple
	if (createReferenceHistos_) fillOne(matchedCand); // fill pt_ref, eta_ref and phi_ref histos for MATCHED reference candidate
      }
      else {
	candBench_.fillOne(matchedCand);   // fill pt, eta phi and charge histos for MATCHED candidate
	//matchCandBench_.fillOne(matchedCand, cand);  // fill delta_x_VS_y histos for matched couple
	matchCandBench_.fillOne(cand, matchedCand, parameterSet);  // fill delta_x_VS_y histos for matched couple
	if (createReferenceHistos_) fillOne(cand); // fill pt_ref, eta_ref and phi_ref histos for MATCHED reference candidate
      }

    }
  }
}

template< class T, class C, class M>
  /*void PFCandidateMonitor::fill(const T& candCollection,
    const C& matchedCandCollection, float& minVal, float& maxVal) {*/
  void PFCandidateMonitor::fill(const T& candCollection,
				const C& matchedCandCollection, float& minVal, float& maxVal,
				const edm::ParameterSet & parameterSet, const M& muonMatchedCandCollection) {

  
  matching_done_ = false;
  if ( createEfficiencyHistos_ ) {
    for( unsigned i=0; i<candCollection.size(); ++i) {
      if( !isInRange(candCollection[i].pt(), candCollection[i].eta(), candCollection[i].phi() ) ) continue;
      fillOne(candCollection[i]); // fill pt_gen, eta_gen and phi_gen histos for UNMATCHED generated candidate

      for( unsigned j=0; j<matchedCandCollection.size(); ++j)  // for DeltaR spectrum
	if (deltaR_) deltaR_->Fill( reco::deltaR( candCollection[i], matchedCandCollection[j] ) ) ;
    }
  }

  std::vector<int> matchIndices;
  //PFB::match( candCollection, matchedCandCollection, matchIndices, matchCharge_, dRMax_ );
  //PFB::match( candCollection, matchedCandCollection, matchIndices, parameterSet, matchCharge_, dRMax_ );
  PFB::match( candCollection, matchedCandCollection, matchIndices, parameterSet, muonMatchedCandCollection, matchCharge_, dRMax_ );
  // now matchIndices[i] stores the j-th closest matched jet 
  matching_done_ = true;

  for (unsigned int i = 0; i < (candCollection).size(); i++) {
     const reco::Candidate& cand = candCollection[i];

    if( !isInRange(cand.pt(), cand.eta(), cand.phi() ) ) continue;
    
    int iMatch = matchIndices[i];
    assert(iMatch< static_cast<int>(matchedCandCollection.size()));

    if( iMatch!=-1 ) {
      const reco::Candidate& matchedCand = matchedCandCollection[ iMatch ];
      if(!isInRange(matchedCand.pt(),matchedCand.eta(),matchedCand.phi() ) ) continue;
      //std::cout <<"PFJet pT " <<cand.pt() <<" eta " <<cand.eta() <<" phi " <<cand.phi() ;
      //std::cout <<"\nmatched genJet pT " <<matchedCand.pt() <<" eta " <<matchedCand.eta() <<" phi " <<matchedCand.phi() <<"\n" <<std::endl ;
      float ptRes = (cand.pt() - matchedCand.pt()) / matchedCand.pt();
      
      if (ptRes > maxVal) maxVal = ptRes;
      if (ptRes < minVal) minVal = ptRes;
 
      if ( !createEfficiencyHistos_ ) {
	candBench_.fillOne(cand);   // fill pt, eta phi and charge histos for MATCHED candidate
	matchCandBench_.fillOne(cand, matchedCand, parameterSet);  // fill delta_x_VS_y histos for matched couple
	if (createReferenceHistos_) fillOne(matchedCand); // fill pt_ref, eta_ref and phi_ref histos for MATCHED reference candidate
      }
      else {
	candBench_.fillOne(matchedCand);   // fill pt, eta phi and charge histos for MATCHED candidate
	matchCandBench_.fillOne(cand, matchedCand, parameterSet);  // fill delta_x_VS_y histos for matched couple
	if (createReferenceHistos_) fillOne(cand); // fill pt_ref, eta_ref and phi_ref histos for MATCHED reference candidate
      }

    }
  }
}
#endif 
