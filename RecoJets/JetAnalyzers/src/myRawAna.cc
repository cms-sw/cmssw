// myRawAna.cc
// Description:  Access raw data, specifically event fragment size per FED
// Author: Jim Hirschauer
// Date:  28 - July - 2009
// 
#include "RecoJets/JetAnalyzers/interface/myRawAna.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <TROOT.h>
#include <TSystem.h>
#include <TFile.h>
#include <TCanvas.h>
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;

#define DEBUG 1
#define MAXJETS 100

// Nothing passed from .cfg yet, but I leave the syntax here for
// future reference.

myRawAna::myRawAna( const ParameterSet & cfg ) {
}


// ************************
// ************************

void myRawAna::beginJob( const EventSetup & ) {

  edm::Service<TFileService> fs;

  fedSize = fs->make<TH2F>( "fedSize", "fedSize", 901, -0.5, 900.5, 20000, 0., 20000. );
  totFedSize = fs->make<TH1F>( "totFedSize", "totFedSize", 200, 0., 20000. );


}

// ************************
// ************************
void myRawAna::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
  
  // **************************************************************
  // ** Access FED Information
  // **************************************************************


  edm::Handle<FEDRawDataCollection> theRaw;
  bool getFed = evt.getByLabel("source", theRaw);
  
  if ( ! getFed ) {
    std::cout << "fedRawData not available" << std::endl;
  } else { // got the fed raw data
    unsigned int totalFEDsize = 0 ; 
    
    // HCAL FEDs are 700-730
    unsigned int fedStart_ = 0;
    unsigned int fedStop_ = 900;
    
    for (unsigned int i=fedStart_; i<=fedStop_; ++i) {
      fedSize->Fill(i,theRaw->FEDData(i).size());
      totalFEDsize += theRaw->FEDData(i).size() ; 
    }
    totFedSize->Fill(totalFEDsize);
  }
  
  
}

// ***********************************
// ***********************************
void myRawAna::endJob() {

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(myRawAna);
