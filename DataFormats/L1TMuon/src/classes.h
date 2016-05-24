#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/L1TMuon/interface/MuonCaloSumFwd.h"
#include "DataFormats/L1TMuon/interface/MuonCaloSum.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrackExtra.h"
#include "DataFormats/L1TMuon/interface/EMTFHitExtra.h"

namespace {
  struct dictionary {
    l1t::MuonCaloSumBxCollection caloSum;
    edm::Wrapper<l1t::MuonCaloSumBxCollection> caloSumWrap;

    l1t::RegionalMuonCandBxCollection regCand;
    edm::Wrapper<l1t::RegionalMuonCandBxCollection> regCandWrap;
   
    l1t::EMTFDaqOutCollection emtfOutput;
    edm::Wrapper<l1t::EMTFDaqOutCollection> emtfOutputWrap;
   
    l1t::EMTFTrackCollection emtfTrack;
    edm::Wrapper<l1t::EMTFTrackCollection> emtfTrackWrap;
   
    l1t::EMTFHitCollection emtfHit;
    edm::Wrapper<l1t::EMTFHitCollection> emtfHitWrap;
   
    l1t::EMTFTrackExtraCollection emtfTrackExtra;
    edm::Wrapper<l1t::EMTFTrackExtraCollection> emtfTrackExtraWrap;
   
    l1t::EMTFHitExtraCollection emtfHitExtra;
    edm::Wrapper<l1t::EMTFHitExtraCollection> emtfHitExtraWrap;
   
  };
}


