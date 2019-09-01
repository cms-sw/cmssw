#ifndef TESTCORRECTION_HH
#define TESTCORRECTION_HH

// -*- C++ -*-
//
// Package:    TestCorrection
// Class:      TestCorrection
// 
/**\class TestCorrection TestCorrection.cc MuonAnalysis/MomentumScaleCalibration/plugins/TestCorrection.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco De Mattia
//         Created:  Thu Sep 11 12:16:00 CEST 2008
//
//

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"

// For the momentum scale correction
#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Muon.h"

#include "TFile.h"
#include "TProfile.h"
#include "TH1F.h"

#include "MuScleFitBase.h"

//
// class decleration
//

class TestCorrection : public edm::EDAnalyzer, MuScleFitBase {
public:
  explicit TestCorrection(const edm::ParameterSet&);
  ~TestCorrection();

private:
  virtual void initialize(const edm::EventSetup&);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {};
  template<typename T>
  std::vector<MuScleFitMuon> fillMuonCollection( const std::vector<T>& tracks )
  {
    std::vector<MuScleFitMuon> muons;
    typename std::vector<T>::const_iterator track;
    for( track = tracks.begin(); track != tracks.end(); ++track ) {
      reco::Particle::LorentzVector mu;
      mu = reco::Particle::LorentzVector(track->px(),track->py(),track->pz(),
					 sqrt(track->p()*track->p() + + 0.011163612));
      
      Double_t hitsTk(0), hitsMuon(0), ptError(0);
      if ( const reco::Muon* myMu = dynamic_cast<const reco::Muon*>(&(*track))  ){
	hitsTk =   myMu->innerTrack()->hitPattern().numberOfValidTrackerHits();
	hitsMuon = myMu->innerTrack()->hitPattern().numberOfValidMuonHits();
	ptError =  myMu->innerTrack()->ptError();
      }
      else if ( const pat::Muon* myMu = dynamic_cast<const pat::Muon*>(&(*track)) ) {
	hitsTk =   myMu->innerTrack()->hitPattern().numberOfValidTrackerHits();
	hitsMuon = myMu->innerTrack()->hitPattern().numberOfValidMuonHits();
	ptError =  myMu->innerTrack()->ptError();
      }
      else if (const reco::Track* myMu = dynamic_cast<const reco::Track*>(&(*track))){
	hitsTk =   myMu->hitPattern().numberOfValidTrackerHits();
	hitsMuon = myMu->hitPattern().numberOfValidMuonHits();
	ptError =  myMu->ptError();
      }
      
      MuScleFitMuon muon(mu,track->charge(),ptError,hitsTk,hitsMuon,false);

    if (debug_>0) {
      std::cout<<"[TestCorrection::fillMuonCollection] after MuScleFitMuon initialization"<<std::endl;
      std::cout<<"  muon = "<<muon<<std::endl;
    }

    muons.push_back(muon);
    }
    return muons;
  }

  lorentzVector correctMuon( const lorentzVector& muon );

  // ----------member data ---------------------------

  // Collections labels
  // ------------------
  TH1F * uncorrectedPt_;
  TProfile * uncorrectedPtVsEta_;
  TH1F * correctedPt_;
  TProfile * correctedPtVsEta_;

  int eventCounter_;

  std::unique_ptr<MomentumScaleCorrector> corrector_;
  std::unique_ptr<ResolutionFunction> resolution_;
  std::unique_ptr<BackgroundFunction> background_;
};

#endif // TESTCORRECTION_HH
