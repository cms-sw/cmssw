#ifndef DataFormats_CTPPSReco_CTPPSFastTrackContainer_H
#define DataFormats_CTPPSReco_CTPPSFastTrackContainer_H

#include "DataFormats/CTPPSReco/interface/CTPPSFastTrack.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <vector>
#include "DataFormats/Common/interface/RefToBase.h" 

namespace edm {
    typedef std::vector<CTPPSFastTrack> CTPPSFastTrackContainer;
} // edm

#endif 

