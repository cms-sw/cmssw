#ifndef L1TMuonEndCap_MicroGMTConverter_h
#define L1TMuonEndCap_MicroGMTConverter_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"


class MicroGMTConverter {
public:
  explicit MicroGMTConverter();
  ~MicroGMTConverter();

  void convert(
      const EMTFTrack& in_track,
      l1t::RegionalMuonCand& out_cand
  ) const;

  void convert_all(
      const EMTFTrackCollection& in_tracks,
      l1t::RegionalMuonCandBxCollection& out_cands
  ) const;

private:
};

#endif
