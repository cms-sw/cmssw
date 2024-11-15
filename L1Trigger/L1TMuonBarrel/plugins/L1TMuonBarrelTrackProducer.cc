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

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTFConfig.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1MuBMTrackFinder.h"

#include <iostream>
#include <iomanip>

class L1TMuonBarrelTrackProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  L1TMuonBarrelTrackProducer(const edm::ParameterSet& pset);

  /// Produce digis out of raw data
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  L1MuBMTrackFinder dtbx_;

  edm::EDPutTokenT<l1t::RegionalMuonCandBxCollection> regionToken_;
  edm::EDPutTokenT<l1t::RegionalMuonCandBxCollection> unsortRegionToken_;
  edm::EDPutTokenT<vector<L1MuBMTrack>> trackToken_;
  edm::EDPutTokenT<vector<L1MuBMTrackSegPhi>> segPhiToken_;
  edm::EDPutTokenT<vector<L1MuBMTrackSegEta>> segEtaToken_;
};

using namespace std;

L1TMuonBarrelTrackProducer::L1TMuonBarrelTrackProducer(const edm::ParameterSet& pset)
    : dtbx_(pset, consumesCollector()) {
  regionToken_ = produces<l1t::RegionalMuonCandBxCollection>("BMTF");
  unsortRegionToken_ = produces<l1t::RegionalMuonCandBxCollection>("UnsortedBMTF");
  trackToken_ = produces<vector<L1MuBMTrack>>("BMTF");
  segPhiToken_ = produces<vector<L1MuBMTrackSegPhi>>("BMTF");
  segEtaToken_ = produces<vector<L1MuBMTrackSegEta>>("BMTF");
  //without clearing before the first call things fail
  dtbx_.clear();
}

void L1TMuonBarrelTrackProducer::produce(edm::Event& e, const edm::EventSetup& c) {
  if (dtbx_.config().Debug(1))
    cout << endl;
  if (dtbx_.config().Debug(1))
    cout << "**** L1MuonBMTFTrigger processing event  ****" << endl;

  dtbx_.run(e, c);

  int ndt = dtbx_.numberOfTracks();
  if (dtbx_.config().Debug(1))
    cout << "Number of muons found by the L1 BBMX TRIGGER : " << ndt << endl;

  ///Muons before muon sorter
  auto tra_product = dtbx_.getcache0();

  ///Muons after muon sorter, for uGMT
  auto vec_product = dtbx_.getcache();

  auto vec_L1MuBMTrack = dtbx_.getcache1();
  auto vec_L1MuBMTrackSegPhi = dtbx_.getcache2();
  auto vec_L1MuBMTrackSegEta = dtbx_.getcache3();

  //for (int ibx = BMTracks.getFirstBX(); ibx  <= BMTracks.getLastBX(); ibx++){
  //cout << "DEBUG:  BMTF size at bx " << ibx << " " << BMTracks.size(ibx) << "\n";
  //}
  e.emplace(unsortRegionToken_, std::move(tra_product));
  e.emplace(regionToken_, std::move(vec_product));
  e.emplace(trackToken_, std::move(vec_L1MuBMTrack));
  e.emplace(segPhiToken_, std::move(vec_L1MuBMTrackSegPhi));
  e.emplace(segEtaToken_, std::move(vec_L1MuBMTrackSegEta));

  dtbx_.clear();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TMuonBarrelTrackProducer);
