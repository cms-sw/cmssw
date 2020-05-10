#ifndef DataFormats_L1TrackTrigger_L1TrackExtra_h
#define DataFormats_L1TrackTrigger_L1TrackExtra_h

#include "DataFormats/L1TrackTrigger/interface/TTTrackExtra.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <vector>

using L1TrackExtra = TTTrackExtra<Ref_Phase2TrackerDigi_>;
using L1TrackExtraCollection = std::vector<L1TrackExtra>;

#endif
