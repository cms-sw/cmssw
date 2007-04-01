//-------------------------------------------------
//
//   Class: DTTrackFinder
//
//   L1 DT Track Finder EDProducer
//
//
//   $Date: 2007/02/27 11:44:00 $
//   $Revision: 1.5 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1Trigger/DTTrackFinder/interface/DTTrackFinder.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h>
#include <DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h>

#include "L1Trigger/DTTrackFinder/src/L1MuDTTFConfig.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTFSetup.h"
#include "L1Trigger/DTTrackFinder/interface/L1MuDTTrackFinder.h"

#include <iostream>
#include <iomanip>

using namespace std;

DTTrackFinder::DTTrackFinder(const edm::ParameterSet & pset) {

  produces<L1MuDTTrackContainer>("DTTF");
  produces<vector<L1MuRegionalCand> >("DT");

  setup1 = new L1MuDTTFSetup(pset);

}

DTTrackFinder::~DTTrackFinder() {

  delete setup1;

}

void DTTrackFinder::produce(edm::Event& e, const edm::EventSetup& c) {

  if ( L1MuDTTFConfig::Debug(1) ) cout << endl;
  if ( L1MuDTTFConfig::Debug(1) ) cout << "**** L1MuonDTTFTrigger processing event  ****" << endl;

  L1MuDTTrackFinder* dtbx = setup1->TrackFinder();
  dtbx->clear();
  dtbx->run(e,c);

  int ndt = dtbx->numberOfTracks();
  if ( L1MuDTTFConfig::Debug(1) ) cout << "Number of muons found by the L1 DTBX TRIGGER : "
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
