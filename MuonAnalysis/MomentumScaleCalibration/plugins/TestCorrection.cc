#ifndef TESTCORRECTION_CC
#define TESTCORRECTION_CC

#include "TestCorrection.h"

#include "TCanvas.h"
#include "TLegend.h"

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
  theMuonLabel_( iConfig.getParameter<edm::InputTag>( "MuonLabel" ) ),
  theMuonType_( iConfig.getParameter<int>( "MuonType" ) ),
  theRootFileName_( iConfig.getUntrackedParameter<string>("OutputFileName") )
{
  //now do what ever initialization is needed
  outputFile_ = new TFile(theRootFileName_.c_str(), "RECREATE");
  outputFile_->cd();
  uncorrectedPt_ = new TH1F("uncorrectedPt", "uncorrected pt", 1000, 0, 100);
  uncorrectedPtVsEta_ = new TProfile("uncorrectedPtVsEta", "uncorrected pt vs eta", 1000, 0, 100, -3., 3.);
  correctedPt_ = new TH1F("correctedPt", "corrected pt", 1000, 0, 100);
  correctedPtVsEta_ = new TProfile("correctedPtVsEta", "corrected pt vs eta", 1000, 0, 100, -3., 3.);
  eventCounter_ = 0;
  // Create the corrector and set the parameters
  corrector_.reset(new MomentumScaleCorrector( iConfig.getUntrackedParameter<string>("CorrectionsIdentifier") ) );
  cout << "corrector_ = " << &*corrector_ << endl;
}


TestCorrection::~TestCorrection()
{
  outputFile_->cd();
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
  outputFile_->Close();

  cout << "Total analyzed events = " << eventCounter_ << endl;
}


//
// member functions
//

// ------------ method called to for each event  ------------
void TestCorrection::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  ++eventCounter_;
  if ( eventCounter_%100 == 0 ) {
    std::cout << "Event number " << eventCounter_ << std::endl;
  }

  // Take the reco-muons, depending on the type selected in the cfg
  // --------------------------------------------------------------

  vector<reco::LeafCandidate> muons;

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

  // Loop on the recMuons
  vector<reco::LeafCandidate>::const_iterator recMuon = muons.begin();
  int muonCount = 0;
  for ( ; recMuon!=muons.end(); ++recMuon, ++muonCount ) {  

    // Fill the histogram with uncorrected pt values
    uncorrectedPt_->Fill(recMuon->pt());
    uncorrectedPtVsEta_->Fill(recMuon->pt(), recMuon->eta());

    // Fill the histogram with corrected pt values
    cout << "correcting muon["<<muonCount<<"] with pt = " << recMuon->pt() << endl;
    double corrPt = (*corrector_)(*recMuon);
    cout << "to pt = " << corrPt << endl;
    correctedPt_->Fill(corrPt);
    correctedPtVsEta_->Fill(corrPt, recMuon->eta());
    // correctedPt_->Fill(recMuon->pt());
  }
}

// ------------ method called once each job just before starting event loop  ------------
void 
TestCorrection::beginJob(const edm::EventSetup&) {}

// ------------ method called once each job just after ending the event loop  ------------
void 
TestCorrection::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(TestCorrection);

#endif // TESTCORRECTION_CC
