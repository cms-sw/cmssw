#ifndef skelsubsys_tselname_tselname_h
#define tselname_h
// -*- C++ -*-
//
// Package:    tselname
// Class:      tselname
// 
/**\class tselname tselname.h skelsubsys/tselname/src/tselname.h

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
#include <TH1.h>
#include "FWCore/TFWLiteSelector/interface/TFWLiteSelector.h"

//A worker processes the events.  When using PROOF there is one Worker per PROOF CPU Node.
struct tselnameWorker {
  tselnameWorker(const TList*, TList&);
  ~tselnameWorker();
  void process( const edm::Event& iEvent );
  void postProcess(TList&);
  //Place histograms, etc that you want to fill here
  //TH1F* h_a;
@example_track  TH1F* h_pt;
};

//Only one Selector is made per job. It gets all the results from each worker.
class tselname : public TFWLiteSelector<tselnameWorker> {
public :
  tselname();
  ~tselname();
  void begin(TList*&);
  void terminate(TList&);
    
private:
    
  tselname(tselname const&);
  tselname operator=(tselname const&);
};
#endif
