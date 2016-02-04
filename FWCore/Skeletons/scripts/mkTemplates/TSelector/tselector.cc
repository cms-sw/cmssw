// -*- C++ -*-
//
// Package:    tselname
// Class:      tselname
// 
/*

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  John Doe
//         Created:  day-mon-xx
// RCS(Id)
//
//


// system include files
#include <memory>
#include <iostream>

#include "TCanvas.h"
// user include files
#include "skelsubsys/tselname/src/tselname.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

@example_track #include "DataFormats/TrackReco/interface/Track.h"
//
// constants shared between the Selector and the Workers
//
@example_track const char* const kPt = "pt";
//const char* const kA = "a";

//===============================================
//A worker processes the events.
// A new worker is created each time the events are processed
//===============================================

// ------------ constructed for each PROOF Node  ------------
// The arguments are
//   fromSelector: these are copies of values set in the selector and sent to all workers
//            out: these are the items which will be passed back to the selector (e.g. histograms)
tselnameWorker::tselnameWorker(const TList* fromSelector, TList& out ) {
  //h_a  = new TH1F( kA , "a"  , 100,  0, 20 );
  //out.Add(h_a);  
@example_track  h_pt = new TH1F(kPt, "P_t",100,0,100);
@example_track  out.Add(h_pt);
}

tselnameWorker::~tselnameWorker()
{
}
// ------------ method called for each event  ------------
void 
tselnameWorker::process( const edm::Event& iEvent ) {
  using namespace edm;
@example_track    using reco::TrackCollection;
  
@example_track    Handle<TrackCollection> tracks;
@example_track    iEvent.getByLabel("ctfWithMaterialTracks",tracks);
@example_track    for(TrackCollection::const_iterator itTrack = tracks->begin();
@example_track        itTrack != tracks->end();                      
@example_track        ++itTrack) {
@example_track       h_pt->Fill(itTrack->pt());  
@example_track    }
  
//  using namespace edmtest;
//  edm::Handle<ThingCollection> hThings;
//  iEvent.getByLabel("Thing",hThings);
//  for ( ThingCollection::const_iterator it = hThings->begin(); 
//        it != hThings->end(); ++it ) {
//    h_a ->Fill( it->a );
//  }
  
}

// ------------ called after processing the events  ------------
// The argument is the same as for the constructor
void 
tselnameWorker::postProcess(TList& out)
{
}


//===============================================
//Only one Selector is made per job. It gets all the results from each worker.
//===============================================
tselname::tselname()
{
}

tselname::~tselname()
{
}

// ------------ called just before all workers are constructed  ------------
void tselname::begin(TList*& toWorkers)
{
}

// ------------ called after all workers have finished  ------------
// The argument 'fromWorkers' contains the accumulated output of all Workers
void tselname::terminate(TList& fromWorkers) {
  using namespace std;
  std::auto_ptr<TCanvas> canvas( new TCanvas() );
//  {
//    TObject* hist = fromWorkers.FindObject(kA);
//    if(0!=hist) {
//      hist->Draw();
//      canvas->SaveAs( "a.jpg" );
//    } else {
//      cout <<"no '"<<kA<<"' histogram"<<endl;
//    }
 // }

@example_track  {
@example_track    TObject* hist = fromWorkers.FindObject(kPt);
@example_track    if(0!=hist) {
@example_track      hist->Draw();
@example_track      canvas->SaveAs( "pt.jpg" );
@example_track    } else {
@example_track      cout <<"no '"<<kPt<<"' histogram"<<endl;
@example_track    }
@example_track  }
  
}
