#ifndef FastSimDataFormats_CTPPSFastSim_CTPPSFastTrackContainer_H
#define FastSimDataFormats_CTPPSFastSim_CTPPSFastTrackContainer_H

#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrack.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <vector>
#include "DataFormats/Common/interface/RefToBase.h"

namespace edm {
  typedef std::vector<CTPPSFastTrack> CTPPSFastTrackContainer;
}  // namespace edm

#endif
