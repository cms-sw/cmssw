#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

#include <TH1F.h>
#include <TROOT.h>
#include <TFile.h>
#include <TSystem.h>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "TStopwatch.h"


int main(int argc, char* argv[])
{
  // ----------------------------------------------------------------------
  // First Part:
  //
  //  * enable FWLite
  //  * book the histograms of interest
  //  * open the input file
  // ----------------------------------------------------------------------

  if ( argc < 4 ) return 0;

  // load framework libraries
  gSystem->Load( "libFWCoreFWLite" );
  FWLiteEnabler::enable();
 
  // book a set of histograms
  fwlite::TFileService fs = fwlite::TFileService(argv[2]);
  TFileDirectory theDir = fs.mkdir("analyzeBasicPat");
  TH1F* jetPt_  = theDir.make<TH1F>("jetPt", "pt",    100,  0.,300.);
  TH1F* jetEta_ = theDir.make<TH1F>("jetEta","eta",   100, -3.,  3.);
  TH1F* jetPhi_ = theDir.make<TH1F>("jetPhi","phi",   100, -5.,  5.);
  TH1F* disc_   = theDir.make<TH1F>("disc", "Discriminant", 100, 0.0, 10.0);
  TH1F* constituentPt_ = theDir.make<TH1F>("constituentPt", "Constituent pT", 100, 0, 300.0);
 
  // open input file (can be located on castor)
  TFile* inFile = TFile::Open( argv[1] );

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
  fwlite::Event ev(inFile);
  TStopwatch timer;
  timer.Start();

  unsigned int nEventsAnalyzed = 0;
  for(ev.toBegin(); !ev.atEnd(); ++ev, ++iEvent){
    edm::EventBase const & event = ev;

    // Handle to the jet collection
    edm::Handle<std::vector<pat::Jet> > jets;
    edm::InputTag jetLabel( argv[3] );
    event.getByLabel(jetLabel, jets);
   
    // loop jet collection and fill histograms
    for(unsigned i=0; i<jets->size(); ++i){
      jetPt_ ->Fill( (*jets)[i].pt()  );
      jetEta_->Fill( (*jets)[i].eta() );
      jetPhi_->Fill( (*jets)[i].phi() );
      reco::SecondaryVertexTagInfo const * svTagInfos = (*jets)[i].tagInfoSecondaryVertex("secondaryVertex");
      if ( svTagInfos != 0 ) {
    if ( svTagInfos->nVertices() > 0 )
      disc_->Fill( svTagInfos->flightDistance(0).value() );
      }
      std::vector<CaloTowerPtr> const & caloConstituents =  (*jets)[i].getCaloConstituents();
      for ( std::vector<CaloTowerPtr>::const_iterator ibegin = caloConstituents.begin(),
          iend = caloConstituents.end(),
          iconstituent = ibegin;
        iconstituent != iend; ++iconstituent ) {
    constituentPt_->Fill( (*iconstituent)->pt() );
      }
    }
    ++nEventsAnalyzed;
  } 
  // close input file
  inFile->Close();

  timer.Stop();

  // print some timing statistics
  Double_t rtime = timer.RealTime();
  Double_t ctime = timer.CpuTime();
  printf("Analyzed events: %d \n",nEventsAnalyzed);
  printf("RealTime=%f seconds, CpuTime=%f seconds\n",rtime,ctime);
  printf("%4.2f events / RealTime second .\n", (double)nEventsAnalyzed/rtime);
  printf("%4.2f events / CpuTime second .\n", (double)nEventsAnalyzed/ctime);
  


  // ----------------------------------------------------------------------
  // Third Part:
  //
  //  * never forget to free the memory of objects you created
  // ----------------------------------------------------------------------

  // in this example there is nothing to do
 
  // that's it!
  return 0;
}

