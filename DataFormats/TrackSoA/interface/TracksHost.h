#ifndef DataFormats_Track_TracksHost_H
#define DataFormats_Track_TracksHost_H

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"

namespace reco {
  using TracksHost = PortableHostMultiCollection<TrackSoA, TrackHitSoA>;
}

#endif  // DataFormats_Track_TracksHost_H
