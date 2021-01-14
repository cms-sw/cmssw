#ifndef DataFormats_L1TrackTrigger_L1Track_h
#define DataFormats_L1TrackTrigger_L1Track_h

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <vector>

using L1Track = TTTrack<Ref_Phase2TrackerDigi_>;
using L1TrackCollection = std::vector<L1Track>;

#endif
