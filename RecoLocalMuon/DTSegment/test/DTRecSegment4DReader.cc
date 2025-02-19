/** \file
 *
 * $Date: 2006/10/26 07:48:21 $
 * $Revision: 1.2 $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/test/DTRecSegment4DReader.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "TFile.h"
#include "TH1F.h"

/* C++ Headers */
#include <string>

using namespace std;
/* ====================================================================== */

/// Constructor
DTRecSegment4DReader::DTRecSegment4DReader(const edm::ParameterSet& pset) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");
 
 // the name of the 4D rec hits collection
  theRecHits4DLabel = pset.getParameter<string>("recHits4DLabel");

  if(debug)
    cout << "[DTRecSegment4DReader] Constructor called" << endl;

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();
  hPositionX = new TH1F("hPositionX","X Position of the Segments",200,-210,210);
}

/// Destructor
DTRecSegment4DReader::~DTRecSegment4DReader() {
  if(debug) 
    cout << "[DTRecSegment4DReader] Destructor called" << endl;

  // Write the histos to file
  theFile->cd();
  hPositionX->Write();
  theFile->Close();
}

/* Operations */ 
void DTRecSegment4DReader::analyze(const edm::Event & event, const
                                 edm::EventSetup& eventSetup) {
  cout << endl<<"--- [DTRecSegment4DReader] Event analysed #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;

  // Get the rechit collection from the event
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  event.getByLabel(theRecHits4DLabel, all4DSegments);

  DTRecSegment4DCollection::const_iterator segment;

  cout<<"Reconstructed segments: "<<endl;
  for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){
    cout<<*segment<<endl;
    hPositionX->Fill( (*segment).localPosition().x());
  }
  cout<<"---"<<endl;


}

DEFINE_FWK_MODULE(DTRecSegment4DReader);
