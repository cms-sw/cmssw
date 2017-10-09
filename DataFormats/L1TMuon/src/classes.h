#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/L1TMuon/interface/MuonCaloSumFwd.h"
#include "DataFormats/L1TMuon/interface/MuonCaloSum.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFRoad.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack2016.h"
#include "DataFormats/L1TMuon/interface/EMTFHit2016.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack2016Extra.h"
#include "DataFormats/L1TMuon/interface/EMTFHit2016Extra.h"

#include "DataFormats/L1TMuon/interface/L1MuBMTrack.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegPhi.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegEta.h"

#include <vector>

namespace {
  struct dictionary {
    l1t::MuonCaloSumBxCollection caloSum;
    edm::Wrapper<l1t::MuonCaloSumBxCollection> caloSumWrap;

    l1t::RegionalMuonCandBxCollection regCand;
    edm::Wrapper<l1t::RegionalMuonCandBxCollection> regCandWrap;

    l1t::EMTFDaqOutCollection emtfOutput;
    edm::Wrapper<l1t::EMTFDaqOutCollection> emtfOutputWrap;

    l1t::EMTFHitCollection emtfHit;
    edm::Wrapper<l1t::EMTFHitCollection> emtfHitWrap;

    l1t::EMTFRoadCollection emtfRoad;
    edm::Wrapper<l1t::EMTFRoadCollection> emtfRoadWrap;

    l1t::EMTFTrackCollection emtfTrack;
    edm::Wrapper<l1t::EMTFTrackCollection> emtfTrackWrap;

    l1t::EMTFTrack2016Collection emtfTrack2016;
    edm::Wrapper<l1t::EMTFTrack2016Collection> emtfTrack2016Wrap;

    l1t::EMTFHit2016Collection emtfHit2016;
    edm::Wrapper<l1t::EMTFHit2016Collection> emtfHit2016Wrap;

    l1t::EMTFTrack2016ExtraCollection emtfTrack2016Extra;
    edm::Wrapper<l1t::EMTFTrack2016ExtraCollection> emtfTrack2016ExtraWrap;

    l1t::EMTFHit2016ExtraCollection emtfHit2016Extra;
    edm::Wrapper<l1t::EMTFHit2016ExtraCollection> emtfHit2016ExtraWrap;

  };
}

namespace L1Trigger_L1TMuonBarrel {
  struct dictionary {
    L1MuBMTrackSegPhi l1mu_trk_ph;
    L1MuBMTrackSegEta l1mu_trk_th;
    L1MuBMTrack       l1mu_trk_tr;
    L1MuBMSecProcId   l1mu_dt_proc;
    L1MuBMTrackSegLoc  l1mu_dt_segloc;
    L1MuBMAddressArray l1mu_dt_addr;

    L1MuBMTrackCollection l1mu_trk_tr_V;
    edm::Wrapper<L1MuBMTrackCollection> l1mu_trk_tr_W;

    L1MuBMTrackSegPhiCollection l1mu_trk_ph_V;
    edm::Wrapper<L1MuBMTrackSegPhiCollection> l1mu_trk_ph_W;

    L1MuBMTrackSegEtaCollection l1mu_trk_th_V;
    edm::Wrapper<L1MuBMTrackSegEtaCollection> l1mu_trk_th_W;
  };
}
