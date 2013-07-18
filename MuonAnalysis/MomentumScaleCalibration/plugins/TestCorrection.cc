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
// $Id: TestCorrection.cc,v 1.13 2012/05/07 10:03:31 emiglior Exp $
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/Candidate/interface/LeafCandidate.h"

// For the momentum scale correction
#include "MuScleFitBase.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/MomentumScaleCorrector.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/ResolutionFunction.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/BackgroundFunction.h"

#include "TFile.h"
#include "TProfile.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TLegend.h"




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
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestCorrection::TestCorrection(const edm::ParameterSet& iConfig) :
  MuScleFitBase( iConfig )
{
  //now do what ever initialization is needed
  TFile * outputFile = new TFile(theRootFileName_.c_str(), "RECREATE");
  theFiles_.push_back(outputFile);
  // outputFile_ = new TFile(theRootFileName_.c_str(), "RECREATE");
  // outputFile_->cd();
  outputFile->cd();
  MuScleFitUtils::resfind = iConfig.getParameter<std::vector<int> >("resfind");
  fillHistoMap(outputFile, 0);
  uncorrectedPt_ = new TH1F("uncorrectedPt", "uncorrected pt", 1000, 0, 100);
  uncorrectedPtVsEta_ = new TProfile("uncorrectedPtVsEta", "uncorrected pt vs eta", 1000, 0, 100, -3., 3.);
  correctedPt_ = new TH1F("correctedPt", "corrected pt", 1000, 0, 100);
  correctedPtVsEta_ = new TProfile("correctedPtVsEta", "corrected pt vs eta", 1000, 0, 100, -3., 3.);
  eventCounter_ = 0;
  // Create the corrector and set the parameters
  corrector_.reset(new MomentumScaleCorrector( iConfig.getUntrackedParameter<std::string>("CorrectionsIdentifier") ) );
  std::cout << "corrector_ = " << &*corrector_ << std::endl;
  resolution_.reset(new ResolutionFunction(iConfig.getUntrackedParameter<std::string>("ResolutionsIdentifier") ) );
  std::cout << "resolution_ = " << &*resolution_ << std::endl;
  background_.reset(new BackgroundFunction(iConfig.getUntrackedParameter<std::string>("BackgroundIdentifier") ) );

  // Initialize the parameters of MuScleFitUtils from those saved in the functions.
  // MuScleFitUtils::parScale = corrector_.getFunction(0)->parameters();
  MuScleFitUtils::resolutionFunction = resolution_->function(0);
  MuScleFitUtils::resolutionFunctionForVec = resolutionFunctionVecService( resolution_->identifiers()[0] );

  MuScleFitUtils::parResol = resolution_->parameters();
}

TestCorrection::~TestCorrection()
{
  theFiles_[0]->cd();
  TCanvas canvas("ptComparison","pt comparison", 1000, 800);
  canvas.cd();
  uncorrectedPt_->GetXaxis()->SetTitle("Pt(GeV)");
  correctedPt_->SetLineColor(kRed);
  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background
  legend->AddEntry(uncorrectedPt_, "original pt");
  legend->AddEntry(correctedPt_, "corrected pt");
  uncorrectedPt_->Draw();
  correctedPt_->Draw("same");
  legend->Draw("same");

  canvas.Write();
  uncorrectedPt_->Write();
  uncorrectedPtVsEta_->Write();
  correctedPt_->Write();
  correctedPtVsEta_->Write();

  writeHistoMap(0);
  theFiles_[0]->Close();

  std::cout << "Total analyzed events = " << eventCounter_ << std::endl;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TestCorrection::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  initialize(iSetup);

  ++eventCounter_;
  if ( eventCounter_%100 == 0 ) {
    std::cout << "Event number " << eventCounter_ << std::endl;
  }

  // Take the reco-muons, depending on the type selected in the cfg
  // --------------------------------------------------------------

  std::vector<reco::LeafCandidate> muons;

  if (theMuonType_==1) { // GlobalMuons
    Handle<reco::MuonCollection> glbMuons;
    iEvent.getByLabel (theMuonLabel_, glbMuons);
    muons = fillMuonCollection(*glbMuons);
  }
  else if (theMuonType_==2) { // StandaloneMuons
    Handle<reco::TrackCollection> saMuons;
    iEvent.getByLabel (theMuonLabel_, saMuons);
    muons = fillMuonCollection(*saMuons);
  }
  else if (theMuonType_==3) { // Tracker tracks
    Handle<reco::TrackCollection> tracks;
    iEvent.getByLabel (theMuonLabel_, tracks);
    muons = fillMuonCollection(*tracks);
  }

  // Find the two muons from the resonance, and set ResFound bool
  // ------------------------------------------------------------
  std::pair <reco::Particle::LorentzVector, reco::Particle::LorentzVector> recMuFromBestRes = 
    MuScleFitUtils::findBestRecoRes (muons);
  if (MuScleFitUtils::ResFound) {
    MuScleFitUtils::SavedPair.push_back( std::make_pair (recMuFromBestRes.first, recMuFromBestRes.second) );
  } else {
    MuScleFitUtils::SavedPair.push_back( std::make_pair (lorentzVector(0.,0.,0.,0.), lorentzVector(0.,0.,0.,0.)) );
  }

  // If resonance found, do the hard work
  // ------------------------------------
  if (MuScleFitUtils::ResFound) {

    // Find weight and reference mass for this muon pair
    // -------------------------------------------------
    // double weight = MuScleFitUtils::computeWeight ((recMu1+recMu2).mass());

    // Use the correction function to correct the pt scale of the muons. Note that this takes into
    // account the corrections from all iterations.
    lorentzVector recMu1 = correctMuon(recMu1);
    lorentzVector recMu2 = correctMuon(recMu2);

    reco::Particle::LorentzVector bestRecRes (recMu1+recMu2);

    //Fill histograms
    //------------------
    mapHisto_["hRecBestMu"]->Fill(recMu1);
    if ((std::abs(recMu1.eta())<2.5) && (recMu1.pt()>2.5)) {
      mapHisto_["hRecBestMu_Acc"]->Fill(recMu1);
    }
    mapHisto_["hRecBestMu"]->Fill(recMu2);
    if ((std::abs(recMu2.eta())<2.5) && (recMu2.pt()>2.5)) {
      mapHisto_["hRecBestMu_Acc"]->Fill(recMu2);
    }
    mapHisto_["hDeltaRecBestMu"]->Fill(recMu1, recMu2);
    
    mapHisto_["hRecBestRes"]->Fill(bestRecRes);
    if ((std::abs(recMu1.eta())<2.5) && (recMu1.pt()>2.5) && (std::abs(recMu2.eta())<2.5) &&  (recMu2.pt()>2.5)){
      mapHisto_["hRecBestRes_Acc"]->Fill(bestRecRes);
      // Fill histogram of Res mass vs muon variable
      mapHisto_["hRecBestResVSMu"]->Fill (recMu1, bestRecRes, -1);
      mapHisto_["hRecBestResVSMu"]->Fill (recMu2, bestRecRes, +1);
    }
  }

  // Loop on the recMuons
  std::vector<reco::LeafCandidate>::const_iterator recMuon = muons.begin();
  int muonCount = 0;
  for ( ; recMuon!=muons.end(); ++recMuon, ++muonCount ) {  

    // Fill the histogram with uncorrected pt values
    uncorrectedPt_->Fill(recMuon->pt());
    uncorrectedPtVsEta_->Fill(recMuon->pt(), recMuon->eta());

    // Fill the histogram with corrected pt values
    std::cout << "correcting muon["<<muonCount<<"] with pt = " << recMuon->pt() << std::endl;
    double corrPt = (*corrector_)(*recMuon);
    std::cout << "to pt = " << corrPt << std::endl;
    correctedPt_->Fill(corrPt);
    correctedPtVsEta_->Fill(corrPt, recMuon->eta());
    // correctedPt_->Fill(recMuon->pt());
  }
}

lorentzVector TestCorrection::correctMuon( const lorentzVector & muon ) {
  double corrPt = corrector_->correct(muon);
  double ptEtaPhiE[4] = {corrPt, muon.Eta(), muon.Phi(), muon.E()};
  return MuScleFitUtils::fromPtEtaPhiToPxPyPz(ptEtaPhiE);
}

// ------------ method called once each job just before starting event loop  ------------
void TestCorrection::initialize(const edm::EventSetup&)
{
  // Read the pdf from root file. They are used by massProb when finding the muon pair, needed
  // for the mass histograms.
  readProbabilityDistributionsFromFile();
}



#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestCorrection);
