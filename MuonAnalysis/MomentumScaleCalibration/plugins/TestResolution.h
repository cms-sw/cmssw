#ifndef TESTRESOLUTION_HH
#define TESTRESOLUTION_HH

// -*- C++ -*-
//
// Package:    TestResolution
// Class:      TestResolution
//
/**\class TestResolution TestResolution.cc MuonAnalysis/MomentumScaleCalibration/plugins/TestResolution.cc

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
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"

// For the momentum scale resolution
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"

#include "TFile.h"
#include "TProfile.h"

//
// class decleration
//

class TestResolution : public edm::EDAnalyzer {
public:
  explicit TestResolution(const edm::ParameterSet&);
  ~TestResolution() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {};
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

  // ----------member data ---------------------------

  // Collections labels
  // ------------------
  edm::InputTag theMuonLabel_;
  edm::EDGetTokenT<reco::MuonCollection> glbMuonsToken_;
  edm::EDGetTokenT<reco::TrackCollection> saMuonsToken_;
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;

  int theMuonType_;
  std::string theRootFileName_;
  TFile * outputFile_;

  TProfile * sigmaPt_;

  int eventCounter_;

  std::unique_ptr<ResolutionFunction> resolutionFunction_;
};

#endif // TESTRESOLUTION_HH
