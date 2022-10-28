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
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

// For the momentum scale resolution
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"

// ROOT includes
#include "TCanvas.h"
#include "TLegend.h"
#include "TFile.h"
#include "TProfile.h"

//
// class decleration
//

class TestResolution : public edm::one::EDAnalyzer<> {
public:
  explicit TestResolution(const edm::ParameterSet&);
  ~TestResolution() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  template <typename T>
  std::vector<reco::LeafCandidate> fillMuonCollection(const std::vector<T>& tracks) {
    std::vector<reco::LeafCandidate> muons;
    typename std::vector<T>::const_iterator track;
    for (track = tracks.begin(); track != tracks.end(); ++track) {
      // Where 0.011163612 is the squared muon mass.
      reco::Particle::LorentzVector mu(
          track->px(), track->py(), track->pz(), sqrt(track->p() * track->p() + 0.011163612));
      reco::LeafCandidate muon(track->charge(), mu);
      // Store muon
      // ----------
      muons.push_back(muon);
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
  TFile* outputFile_;

  TProfile* sigmaPt_;

  int eventCounter_;

  std::unique_ptr<ResolutionFunction> resolutionFunction_;
};

//
// constructors and destructor
//
TestResolution::TestResolution(const edm::ParameterSet& iConfig)
    : theMuonLabel_(iConfig.getParameter<edm::InputTag>("MuonLabel")),
      glbMuonsToken_(mayConsume<reco::MuonCollection>(theMuonLabel_)),
      saMuonsToken_(mayConsume<reco::TrackCollection>(theMuonLabel_)),
      tracksToken_(mayConsume<reco::TrackCollection>(theMuonLabel_)),
      theMuonType_(iConfig.getParameter<int>("MuonType")),
      theRootFileName_(iConfig.getUntrackedParameter<std::string>("OutputFileName")) {
  //now do what ever initialization is needed
  outputFile_ = new TFile(theRootFileName_.c_str(), "RECREATE");
  outputFile_->cd();
  sigmaPt_ = new TProfile("sigmaPtOverPt", "sigmaPt/Pt vs muon Pt", 1000, 0, 100);
  eventCounter_ = 0;
  // Create the corrector and set the parameters
  resolutionFunction_ =
      std::make_unique<ResolutionFunction>(iConfig.getUntrackedParameter<std::string>("ResolutionsIdentifier"));
  std::cout << "resolutionFunction_ = " << &*resolutionFunction_ << std::endl;
}

TestResolution::~TestResolution() {
  outputFile_->cd();
  TCanvas canvas("sigmaPtOverPt", "sigmaPt/Pt vs muon Pt", 1000, 800);
  canvas.cd();
  sigmaPt_->GetXaxis()->SetTitle("Pt(GeV)");
  //   TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  //   legend->SetTextSize(0.02);
  //   legend->SetFillColor(0); // Have a white background
  //   legend->AddEntry(uncorrectedPt_, "original pt");
  //   legend->AddEntry(correctedPt_, "corrected pt");
  sigmaPt_->Draw();
  //   legend->Draw("same");

  canvas.Write();
  sigmaPt_->Write();
  outputFile_->Close();

  std::cout << "Total analyzed events = " << eventCounter_ << std::endl;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TestResolution::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  ++eventCounter_;
  if (eventCounter_ % 100 == 0) {
    std::cout << "Event number " << eventCounter_ << std::endl;
  }

  // Take the reco-muons, depending on the type selected in the cfg
  // --------------------------------------------------------------

  std::vector<reco::LeafCandidate> muons;

  if (theMuonType_ == 1) {  // GlobalMuons
    Handle<reco::MuonCollection> glbMuons;
    iEvent.getByToken(glbMuonsToken_, glbMuons);
    muons = fillMuonCollection(*glbMuons);
  } else if (theMuonType_ == 2) {  // StandaloneMuons
    Handle<reco::TrackCollection> saMuons;
    iEvent.getByToken(saMuonsToken_, saMuons);
    muons = fillMuonCollection(*saMuons);
  } else if (theMuonType_ == 3) {  // Tracker tracks
    Handle<reco::TrackCollection> tracks;
    iEvent.getByToken(tracksToken_, tracks);
    muons = fillMuonCollection(*tracks);
  }

  // Loop on the recMuons
  std::vector<reco::LeafCandidate>::const_iterator recMuon = muons.begin();
  for (; recMuon != muons.end(); ++recMuon) {
    // Fill the histogram with uncorrected pt values
    sigmaPt_->Fill(resolutionFunction_->sigmaPt(*recMuon, 0), recMuon->pt());
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestResolution);
