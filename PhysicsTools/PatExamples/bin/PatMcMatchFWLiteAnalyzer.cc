#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <TH2F.h> // Use the correct histograms
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Muon.h" // Include the examined data formats
#include "DataFormats/PatCandidates/interface/Jet.h"  // Include the examined data formats
#include "FWCore/FWLite/interface/FWLiteEnabler.h"


int main(int argc, char* argv[]) 
{
  // ----------------------------------------------------------------------
  // First Part: 
  //
  //  * enable FWLite 
  //  * book the histograms of interest 
  //  * open the input file
  // ----------------------------------------------------------------------

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();
  
  // book a set of histograms
  TH2F* muonPt_  = new TH2F( "muonPt", "Muon Pt", 60, 0., 300., 60, 0., 300. ); // 2-D histo for muon Pt
  muonPt_->SetXTitle( "gen." );
  muonPt_->SetYTitle( "reco." );
  TH2F* jetPt_   = new TH2F( "jetPt", "Jet Pt", 100, 0., 500., 100, 0., 500. ); // 2-D histo for jet Pt
  jetPt_->SetXTitle( "gen." );
  jetPt_->SetYTitle( "reco." );
  
  // open input file (can be located on castor)
  TFile* inFile = TFile::Open( "file:edmPatMcMatch.root" ); // Adapt the input file name

  // ----------------------------------------------------------------------
  // Second Part: 
  //
  //  * loop the events in the input file 
  //  * receive the collections of interest via fwlite::Handle
  //  * fill the histograms
  //  * after the loop close the input file
  // ----------------------------------------------------------------------

  // loop the events
  unsigned int iEvent=0;
  fwlite::Event event(inFile);
  for(event.toBegin(); !event.atEnd(); ++event, ++iEvent){
    // break loop after end of file is reached 
    // or after 1000 events have been processed
    if( iEvent==1000 ) break;
    
    // simple event counter
    if(iEvent>0 && iEvent%25==0){ // Reduce print-out
      std::cout << "  processing event: " << iEvent << std::endl;
    }

    // fwlite::Handle to the muon collection
    fwlite::Handle<std::vector<pat::Muon> > muons; // Access the muon collection
    muons.getByLabel(event, "cleanLayer1Muons");
    // fwlite::Handle to the muon collection
    fwlite::Handle<std::vector<pat::Jet> > jets; // Access the jet collection
    jets.getByLabel(event, "cleanLayer1Jets");
    
    // loop muon collection and fill histograms
    for(unsigned i=0; i<muons->size(); ++i){
      const reco::GenParticle * genMuon = (*muons)[i].genParticle(); // Get the matched generator muon
      if ( genMuon ) {                                               // Check for valid reference
        muonPt_->Fill( genMuon->pt(), (*muons)[i].pt() );            // Fill 2-D histo
      }
    }
    // loop jet collection and fill histograms
    for(unsigned i=0; i<jets->size(); ++i){
      const reco::GenJet * genJet = (*jets)[i].genJet(); // Get the matched generator jet
      if ( genJet ) {                                    // Check for valid reference
        jetPt_->Fill( genJet->pt(), (*jets)[i].pt() );   // Fill 2-D histo
      }
    }
  }  
  // close input file
  inFile->Close();

  // ----------------------------------------------------------------------
  // Third Part: 
  //
  //  * open the output file 
  //  * write the histograms to the output file
  //  * close the output file
  // ----------------------------------------------------------------------
  
  //open output file
  TFile outFile( "rootPatMcMatch.root", "recreate" ); // Adapt the output file name
  outFile.mkdir("analyzeMcMatchPat");                 // Adapt output file according to modifications
  outFile.cd("analyzeMcMatchPat");
  muonPt_->Write( );
  jetPt_->Write( );
  outFile.Close();
  
  // ----------------------------------------------------------------------
  // Fourth Part: 
  //
  //  * never forgett to free the memory of the histograms
  // ----------------------------------------------------------------------

  // free allocated space
  delete muonPt_; // Delete the muon histo
  delete jetPt_;  // Delete the jet histo
  
  // that's it!
  return 0;
}
