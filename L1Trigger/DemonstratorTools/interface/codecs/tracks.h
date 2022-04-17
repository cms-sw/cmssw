
#ifndef L1Trigger_DemonstratorTools_codecs_tracks_h
#define L1Trigger_DemonstratorTools_codecs_tracks_h

#include <array>
#include <sstream>
#include <vector>

#include "ap_int.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

#include "L1Trigger/DemonstratorTools/interface/BoardData.h"

namespace l1t::demo::codecs {

  ap_uint<96> encodeTrack(const TTTrack_TrackWord& t);

  // Encodes track collection onto 18 'logical' output links (2x9 eta-phi sectors; first 9 negative eta)
  std::array<std::vector<ap_uint<64>>, 18> encodeTracks(const edm::View<TTTrack<Ref_Phase2TrackerDigi_>>&,
                                                        int debug = 0);

}  // namespace l1t::demo::codecs

#endif
