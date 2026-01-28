#ifndef DataFormats_TrackSoA_interface_TracksHost_H
#define DataFormats_TrackSoA_interface_TracksHost_H

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TrackSoA/interface/TracksSoA.h"

namespace reco {
  using TracksHost = PortableHostCollection<TrackBlocks>;
}

#endif  // DataFormats_TrackSoA_interface_TracksHost_H
