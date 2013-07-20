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
// $Id: TestCorrection.h,v 1.10 2012/12/20 16:09:29 emiglior Exp $
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
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"

// For the momentum scale correction
#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"

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
  std::vector<reco::LeafCandidate> fillMuonCollection (const std::vector<T>& tracks) {
    std::vector<reco::LeafCandidate> muons;
    typename std::vector<T>::const_iterator track;
    for (track = tracks.begin(); track != tracks.end(); ++track){
      // Where 0.011163612 is the squared muon mass.
      reco::Particle::LorentzVector mu(track->px(),track->py(),track->pz(),
				       sqrt(track->p()*track->p() + 0.011163612));
      reco::LeafCandidate muon(track->charge(),mu);
      // Store muon
      // ----------
      muons.push_back (muon);
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

  std::auto_ptr<MomentumScaleCorrector> corrector_;
  std::auto_ptr<ResolutionFunction> resolution_;
  std::auto_ptr<BackgroundFunction> background_;
};

#endif // TESTCORRECTION_HH
