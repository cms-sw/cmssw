/***For CTPPS FastSim*/
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrack.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastTrackContainer.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHit.h"
#include "FastSimDataFormats/CTPPSFastSim/interface/CTPPSFastRecHitContainer.h"
/***/


#include <vector>

namespace FastSimDataFormats_CTPPSFastSim {
  struct dictionary {


    //--- fastsim objects
    CTPPSFastTrack xxxxt;
    edm::CTPPSFastTrackContainer sxxxxt;
    edm::Wrapper<edm::CTPPSFastTrackContainer> dummy1;
    std::vector<const CTPPSFastTrack*> dummy2;

    CTPPSFastRecHit xxxxr;
    edm::CTPPSFastRecHitContainer sxxxxr;
    edm::Wrapper<edm::CTPPSFastRecHitContainer> dummy3;
    std::vector<const CTPPSFastRecHit*> dummy4;


  };
}
