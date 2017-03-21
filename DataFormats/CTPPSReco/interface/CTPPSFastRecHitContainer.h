#ifndef DataFormats_CTPPSReco_CTPPSFastRecHitContainer_H
#define DataFormats_CTPPSReco_CTPPSFastRecHitContainer_H

#include "DataFormats/CTPPSReco/interface/CTPPSFastRecHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <vector>
#include "DataFormats/Common/interface/RefToBase.h" 

namespace edm {
    typedef std::vector<CTPPSFastRecHit> CTPPSFastRecHitContainer;
} // edm

#endif 

