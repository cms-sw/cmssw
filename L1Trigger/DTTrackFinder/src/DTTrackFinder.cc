//-------------------------------------------------
//
//   Class: DTTrackFinder
//
//   L1 DT Track Finder EDProducer
//
//
//   $Date: 2006/06/01 00:00:00 $
//   $Revision: 1.1 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1Trigger/DTTrackFinder/interface/DTTrackFinder.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuRegionalCand.h>

using namespace edm;
using namespace std;

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFSetup.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackFinder.h"

#include <iostream>
#include <iomanip>

int ev_=-1;

DTTrackFinder::DTTrackFinder(const ParameterSet & pset) {

  string lutdir_ = pset.getUntrackedParameter<string>("lutdir","../parameters/");
  setenv("DTTF_DATA_PATH",lutdir_.c_str(),1);

  produces<L1MuDTTrackContainer>("DTTF");
  produces<vector<L1MuRegionalCand> >("DT");

  setup1 = new L1MuDTTFSetup();

}

DTTrackFinder::~DTTrackFinder() {

  delete setup1;

}

void DTTrackFinder::produce(Event& e, const EventSetup& c) {

  cout << endl;
  cout << "**** L1MuonDTTFTrigger processing event  ****" << endl;
  ev_++;
  cout << "EVENT   " << ev_ << endl;

  L1MuDTTrackFinder* dtbx = setup1->TrackFinder();
  dtbx->clear();
  dtbx->run(e);

  int ndt = dtbx->numberOfTracks();
  cout << "Number of muons found by the L1 DTBX TRIGGER : "
       << ndt << endl;

  auto_ptr<L1MuDTTrackContainer> tra_product(new L1MuDTTrackContainer);
  auto_ptr<vector<L1MuRegionalCand> >
                                 vec_product(new vector<L1MuRegionalCand>);

  vector<L1MuRegionalCand>& dtTracks = dtbx->getcache();
  tra_product->setContainer(dtTracks);
  *vec_product = dtTracks; 

  e.put(tra_product,"DTTF");
  e.put(vec_product,"DT");

}
