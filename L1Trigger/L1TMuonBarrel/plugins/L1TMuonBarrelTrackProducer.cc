//-------------------------------------------------
//
//   Class: L1TMuonBarrelTrackProducer
//
//   L1 BM Track Finder EDProducer
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//   Modified :
//   G. Flouris               U Ioannina
//--------------------------------------------------

#include "L1TMuonBarrelTrackProducer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include <FWCore/Framework/interface/ConsumesCollector.h>


#include "L1Trigger/L1TMuonBarrel/src/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTFSetup.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrackFinder.h"

#include <iostream>
#include <iomanip>

using namespace std;

L1TMuonBarrelTrackProducer::L1TMuonBarrelTrackProducer(const edm::ParameterSet & pset) {
  m_ps = &pset;


  produces<l1t::RegionalMuonCandBxCollection>("BMTF");
  produces<l1t::RegionalMuonCandBxCollection>("UnsortedBMTF");
  produces<vector<L1MuBMTrack> >("BMTF");
  produces<vector<L1MuBMTrackSegPhi> >("BMTF");
  produces<vector<L1MuBMTrackSegEta> >("BMTF");

  usesResource("L1TMuonBarrelTrackProducer");
  setup1 = new L1MuBMTFSetup(*m_ps,consumesCollector());


}

L1TMuonBarrelTrackProducer::~L1TMuonBarrelTrackProducer() {

  delete setup1;
}

void L1TMuonBarrelTrackProducer::produce(edm::Event& e, const edm::EventSetup& c) {


  if ( L1MuBMTFConfig::Debug(1) ) cout << endl;
  if ( L1MuBMTFConfig::Debug(1) ) cout << "**** L1MuonBMTFTrigger processing event  ****" << endl;

  L1MuBMTrackFinder* dtbx = setup1->TrackFinder();
  dtbx->clear();

  dtbx->run(e,c);

  int ndt = dtbx->numberOfTracks();
  if ( L1MuBMTFConfig::Debug(1) ) cout << "Number of muons found by the L1 BBMX TRIGGER : "
                                       << ndt << endl;

  std::unique_ptr<l1t::RegionalMuonCandBxCollection> tra_product(new l1t::RegionalMuonCandBxCollection);
  std::unique_ptr<l1t::RegionalMuonCandBxCollection> vec_product(new l1t::RegionalMuonCandBxCollection);
  unique_ptr<vector<L1MuBMTrack> > vec_L1MuBMTrack(new vector<L1MuBMTrack>);
  unique_ptr<vector<L1MuBMTrackSegPhi> > vec_L1MuBMTrackSegPhi(new vector<L1MuBMTrackSegPhi>);
  unique_ptr<vector<L1MuBMTrackSegEta> > vec_L1MuBMTrackSegEta(new vector<L1MuBMTrackSegEta>);

  ///Muons before muon sorter
  l1t::RegionalMuonCandBxCollection  dtTracks = dtbx->getcache0();
  *tra_product = dtTracks;

  ///Muons after muon sorter, for uGMT
  l1t::RegionalMuonCandBxCollection BMTracks = dtbx->getcache();
  *vec_product = BMTracks;

  *vec_L1MuBMTrack = dtbx->getcache1();
  *vec_L1MuBMTrackSegPhi = dtbx->getcache2();
  *vec_L1MuBMTrackSegEta = dtbx->getcache3();

  //for (int ibx = BMTracks.getFirstBX(); ibx  <= BMTracks.getLastBX(); ibx++){
  //cout << "DEBUG:  BMTF size at bx " << ibx << " " << BMTracks.size(ibx) << "\n";
  //}
  e.put(std::move(tra_product),"UnsortedBMTF");
  e.put(std::move(vec_product),"BMTF");
  e.put(std::move(vec_L1MuBMTrack),"BMTF");
  e.put(std::move(vec_L1MuBMTrackSegPhi),"BMTF");
  e.put(std::move(vec_L1MuBMTrackSegEta),"BMTF");
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonBarrelTrackProducer);

