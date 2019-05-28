#ifndef FastSimDataFormats_CTPPSFastSim_CTPPSFastRecHitContainer_H
#define FastSimDataFormats_CTPPSFastSim_CTPPSFastRecHitContainer_H

//FastSimDataFormats/CTPPSFastSim

#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHit.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <vector>
#include "DataFormats/Common/interface/RefToBase.h"

namespace edm {
  typedef std::vector<CTPPSFastRecHit> CTPPSFastRecHitContainer;
}  // namespace edm

#endif
