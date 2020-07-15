// -*- C++ -*-
//
// Package:    __subsys__/__pkgname__
// Class:      __class__
//
/*

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  __author__
//         Created:  __date__
//
//

// system include files
#include <memory>
#include <iostream>

#include "TCanvas.h"
// user include files
#include "__subsys__/__pkgname__/plugins/__class__.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

@example_track#include "DataFormats/TrackReco/interface/Track.h"
//
// constants shared between the Selector and the Workers
//
@example_trackconst char* const kPt = "pt";
//const char* const kA = "a";

//===============================================
//A worker processes the events.
// A new worker is created each time the events are processed
//===============================================

// ------------ constructed for each PROOF Node  ------------
// The arguments are
//   fromSelector: these are copies of values set in the selector and sent to all workers
//            out: these are the items which will be passed back to the selector (e.g. histograms)
__class__Worker::__class__Worker(const TList* fromSelector, TList& out) {
  //h_a  = new TH1F( kA , "a"  , 100,  0, 20 );
  //out.Add(h_a);
@example_track  h_pt = new TH1F(kPt, "P_t", 100, 0, 100);
@example_track  out.Add(h_pt);
}

__class__Worker::~__class__Worker() {
}
// ------------ method called for each event  ------------
void __class__Worker::process(const edm::Event& iEvent) {
  using namespace edm;
@example_track  using reco::TrackCollection;
@example_track
@example_track  Handle<TrackCollection> tracks;
@example_track  iEvent.getByLabel("ctfWithMaterialTracks", tracks);
@example_track  for (const auto& track : *tracks) {
@example_track    h_pt->Fill(itTrack->pt());
@example_track  }

  //using namespace edmtest;
  //edm::Handle<ThingCollection> hThings;
  //iEvent.getByLabel("Thing", hThings);
  //for (const auto& thing : *hThings) {
  //  h_a ->Fill(it->a);
  //}
}

// ------------ called after processing the events  ------------
// The argument is the same as for the constructor
void __class__Worker::postProcess(TList& out) {
}

//===============================================
//Only one Selector is made per job. It gets all the results from each worker.
//===============================================
__class__::__class__() {
}

__class__::~__class__() {
}

// ------------ called just before all workers are constructed  ------------
void __class__::begin(TList*& toWorkers) {
}

// ------------ called after all workers have finished  ------------
// The argument 'fromWorkers' contains the accumulated output of all Workers
void __class__::terminate(TList& fromWorkers) {
  using namespace std;
  auto canvas = std::make_unique<TCanvas>();
  //{
  //  TObject* hist = fromWorkers.FindObject(kA);
  //  if (nullptr != hist) {
  //    hist->Draw();
  //    canvas->SaveAs("a.jpg");
  //  } else {
  //    cout << "no '" << kA << "' histogram" << endl;
  //  }
  //
@example_track
@example_track  {
@example_track    TObject* hist = fromWorkers.FindObject(kPt);
@example_track    if (nullptr != hist) {
@example_track      hist->Draw();
@example_track      canvas->SaveAs("pt.jpg");
@example_track    } else {
@example_track      cout << "no '" << kPt << "' histogram" << endl;
@example_track    }
@example_track  }
}
