/** \file
 *
 * $Date: 2006/10/26 07:48:21 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 */

/* This Class Header */
#include "RecoLocalMuon/DTSegment/test/DTRecSegment2DReader.h"

/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"

#include "TFile.h"
#include "TH1F.h"

/* C++ Headers */
#include <string>

using namespace std;
/* ====================================================================== */

/// Constructor
DTRecSegment2DReader::DTRecSegment2DReader(const edm::ParameterSet& pset) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");
 
 // the name of the 2D rec hits collection
  theRecHits2DLabel = pset.getParameter<string>("recHits2DLabel");

  if(debug)
    cout << "[DTRecSegment2DReader] Constructor called" << endl;

  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");
  theFile->cd();
  hPositionX = new TH1F("hPositionX","X Position of the Segments",200,-210,210);
}

/// Destructor
DTRecSegment2DReader::~DTRecSegment2DReader() {
  if(debug) 
    cout << "[DTRecSegment2DReader] Destructor called" << endl;

  // Write the histos to file
  theFile->cd();
  hPositionX->Write();
  theFile->Close();
}

/* Operations */ 
void DTRecSegment2DReader::analyze(const edm::Event & event, const
                                 edm::EventSetup& eventSetup) {
  cout << endl<<"--- [DTRecSegment2DReader] Event analysed #Run: " << event.id().run()
       << " #Event: " << event.id().event() << endl;

  // Get the rechit collection from the event
  edm::Handle<DTRecSegment2DCollection> all2DSegments;
  event.getByLabel(theRecHits2DLabel, all2DSegments);

  DTRecSegment2DCollection::const_iterator segment;

  cout<<"Reconstructed segments: "<<endl;
  for (segment = all2DSegments->begin(); segment != all2DSegments->end(); ++segment){
    cout<<*segment<<endl;
    hPositionX->Fill( (*segment).localPosition().x());
  }
  cout<<"---"<<endl;


}

DEFINE_FWK_MODULE(DTRecSegment2DReader);
