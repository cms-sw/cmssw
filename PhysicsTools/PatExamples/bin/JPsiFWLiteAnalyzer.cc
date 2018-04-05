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

#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
#include "FWCore/FWLite/interface/FWLiteEnabler.h"

using namespace std;

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
    
  // open input file (can be located on castor)
  TFile* inFile = TFile::Open( "file:jpsi.root" );

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
    if(iEvent>0 && iEvent%1==0){
      std::cout << "  processing event: " << iEvent << std::endl;
    }

    // fwlite::Handle to to jpsi collection
    fwlite::Handle<std::vector<pat::CompositeCandidate> > jpsis;
    jpsis.getByLabel(event, "patJPsiCandidates");
    
    // loop jpsi collection and fill histograms
    for(unsigned i=0; i<jpsis->size(); ++i){
      cout << "jpsi " << i << ", mass = " << jpsis->at(i).mass() << ", dR = " << jpsis->at(i).userFloat("dR") << endl;
    }
  }  
  // close input file
  inFile->Close();
  
  // that's it!
  return 0;
}
