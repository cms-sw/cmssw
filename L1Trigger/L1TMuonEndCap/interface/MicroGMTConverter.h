#ifndef L1TMuonEndCap_MicroGMTConverter_h
#define L1TMuonEndCap_MicroGMTConverter_h

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"

class MicroGMTConverter {
public:
  explicit MicroGMTConverter();
  ~MicroGMTConverter();

  void convert(const int global_event_BX, const EMTFTrack& in_track, l1t::RegionalMuonCand& out_cand) const;

  void convert_all(const edm::Event& iEvent,
                   const EMTFTrackCollection& in_tracks,
                   l1t::RegionalMuonCandBxCollection& out_cands) const;

private:
};  // End class MicroGMTConverter

namespace emtf {
  void sort_uGMT_muons(l1t::RegionalMuonCandBxCollection& cands);
}

#endif
