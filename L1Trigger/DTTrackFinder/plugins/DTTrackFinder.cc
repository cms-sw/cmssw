//-------------------------------------------------
//
//   Class: DTTrackFinder
//
//   L1 DT Track Finder EDProducer
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "DTTrackFinder.h"

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

DTTrackFinder::DTTrackFinder(const edm::ParameterSet& pset) {
  produces<L1MuDTTrackContainer>("DTTF");
  produces<vector<L1MuRegionalCand> >("DT");

  setup1 = new L1MuDTTFSetup(pset, consumesCollector());
  usesResource("DTTrackFinder");
}

DTTrackFinder::~DTTrackFinder() { delete setup1; }

void DTTrackFinder::produce(edm::Event& e, const edm::EventSetup& c) {
  if (L1MuDTTFConfig::Debug(1))
    cout << endl;
  if (L1MuDTTFConfig::Debug(1))
    cout << "**** L1MuonDTTFTrigger processing event  ****" << endl;

  L1MuDTTrackFinder* dtbx = setup1->TrackFinder();
  dtbx->clear();
  dtbx->run(e, c);

  int ndt = dtbx->numberOfTracks();
  if (L1MuDTTFConfig::Debug(1))
    cout << "Number of muons found by the L1 DTBX TRIGGER : " << ndt << endl;

  unique_ptr<L1MuDTTrackContainer> tra_product(new L1MuDTTrackContainer);
  unique_ptr<vector<L1MuRegionalCand> > vec_product(new vector<L1MuRegionalCand>);

  vector<L1MuDTTrackCand>& dtTracks = dtbx->getcache0();
  tra_product->setContainer(dtTracks);
  vector<L1MuRegionalCand>& DTTracks = dtbx->getcache();
  *vec_product = DTTracks;

  e.put(std::move(tra_product), "DTTF");
  e.put(std::move(vec_product), "DT");
}
