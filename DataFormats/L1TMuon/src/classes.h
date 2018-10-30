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
#include "DataFormats/L1TMuon/interface/CPPFDigi.h"

#include "DataFormats/L1TMuon/interface/L1MuBMTrack.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegPhi.h"
#include "DataFormats/L1TMuon/interface/L1MuBMTrackSegEta.h"
#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "DataFormats/L1TMuon/interface/L1MuKBMTrack.h"

#include <vector>

namespace DataFormats_L1TMuon {
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
   
    l1t::CPPFDigiCollection cppfDigi;
    edm::Wrapper<l1t::CPPFDigiCollection> cppfDigiWrap;

    l1t::RegionalMuonCandRef rmcr;
    std::vector<l1t::RegionalMuonCandRef> v_rmcr;
    l1t::RegionalMuonCandRefBxCollection rmcrbxc;
    edm::Wrapper<l1t::RegionalMuonCandRefBxCollection> w_rmcrbxc;
    l1t::RegionalMuonCandRefPair rmcrp;
    std::vector<l1t::RegionalMuonCandRefPair> v_rmcrp;
    l1t::RegionalMuonCandRefPairBxCollection rmcrpc;
    edm::Wrapper<l1t::RegionalMuonCandRefPairBxCollection> w_rmcrpc;
  };
}

namespace L1Trigger_L1TMuonBarrel {
  struct dictionary {
    L1MuKBMTCombinedStub l1mu_stub_comp;
    L1MuBMTrackSegPhi l1mu_trk_ph;
    L1MuBMTrackSegEta l1mu_trk_th;
    L1MuBMTrack       l1mu_trk_tr;
    L1MuKBMTrack       l1muk_trk_tr;
    L1MuBMSecProcId   l1mu_dt_proc;
    L1MuBMTrackSegLoc  l1mu_dt_segloc;
    L1MuBMAddressArray l1mu_dt_addr;

    L1MuBMTrackCollection l1mu_trk_tr_V;
    edm::Wrapper<L1MuBMTrackCollection> l1mu_trk_tr_W;

    L1MuKBMTrackBxCollection l1muk_trk_tr_V;
    edm::Wrapper<L1MuKBMTrackBxCollection> l1muk_trk_tr_W;

    L1MuKBMTrackCollection l1muk_trk_tr_V2;
    edm::Wrapper<L1MuKBMTrackCollection> l1muk_trk_tr_W2;

    L1MuKBMTCombinedStubCollection l1mu_stub_comb_V;
    edm::Wrapper<L1MuKBMTCombinedStubCollection> l1mu_stub_comb_W;
    edm::Ref<L1MuKBMTCombinedStubCollection > l1mu_stub_comb_R;
    edm::Wrapper<std::vector<edm::Ref<L1MuKBMTCombinedStubCollection > > > l1mu_stub_comb_RW;


    L1MuBMTrackSegPhiCollection l1mu_trk_ph_V;
    edm::Wrapper<L1MuBMTrackSegPhiCollection> l1mu_trk_ph_W;

    L1MuBMTrackSegEtaCollection l1mu_trk_th_V;
    edm::Wrapper<L1MuBMTrackSegEtaCollection> l1mu_trk_th_W;
  };
}

