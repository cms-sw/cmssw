#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/L1TMuon/interface/MuonCaloSumFwd.h"
#include "DataFormats/L1TMuon/interface/MuonCaloSum.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack2016.h"
#include "DataFormats/L1TMuon/interface/EMTFHit2016.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack2016Extra.h"
#include "DataFormats/L1TMuon/interface/EMTFHit2016Extra.h"

#include <vector>

namespace {
  struct dictionary {
    l1t::MuonCaloSumBxCollection caloSum;
    edm::Wrapper<l1t::MuonCaloSumBxCollection> caloSumWrap;

    l1t::RegionalMuonCandBxCollection regCand;
    edm::Wrapper<l1t::RegionalMuonCandBxCollection> regCandWrap;
   
    l1t::EMTFDaqOutCollection emtfOutput;
    edm::Wrapper<l1t::EMTFDaqOutCollection> emtfOutputWrap;
   
    l1t::EMTFTrack2016Collection emtfTrack;
    edm::Wrapper<l1t::EMTFTrack2016Collection> emtfTrackWrap;
   
    l1t::EMTFHit2016Collection emtfHit;
    edm::Wrapper<l1t::EMTFHit2016Collection> emtfHitWrap;
   
    l1t::EMTFTrack2016ExtraCollection emtfTrackExtra;
    edm::Wrapper<l1t::EMTFTrack2016ExtraCollection> emtfTrackExtraWrap;
   
    l1t::EMTFHit2016ExtraCollection emtfHitExtra;
    edm::Wrapper<l1t::EMTFHit2016ExtraCollection> emtfHitExtraWrap;
   
  };
}


