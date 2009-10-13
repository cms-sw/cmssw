#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Alignment/interface/AlignmentClusterFlag.h"
#include "DataFormats/Alignment/interface/AliClusterValueMap.h"
#include <vector>
#include <map>
#include "DataFormats/Alignment/interface/SiStripLaserRecHit2D.h"
#include "DataFormats/Alignment/interface/TkLasBeam.h"
#include "DataFormats/Alignment/interface/TkFittedLasBeam.h"


namespace {
  struct dictionary {
    //////////////////////////////////////////////
    // for LAS interface to track-based alignment
    //////////////////////////////////////////////
    // TkLasBeam beam1; // not needed since not templated
    // edm::Wrapper<TkLasBeam> beam2; // not needed since not an EDProduct?
    TkLasBeamCollection beamCollection1;
    edm::Wrapper<TkLasBeamCollection> beamCollection2;

    // TkFittedLasBeam fitBeam1; // not needed since not templated?
    TkFittedLasBeamCollection fitBeamCollection1;
    edm::Wrapper<TkFittedLasBeamCollection> fitBeamCollection2;


    //////////////////////////////////////////////
    // for AlCaSkim
    //////////////////////////////////////////////
    // not needed (not instance of template):
    // AlignmentClusterFlag            ahf;
    // edm::Wrapper<AlignmentClusterFlag> wahf; // not needed since not an EDProduct?
    AliClusterValueMap                 ahvm1;
    edm::Wrapper<AliClusterValueMap>   wahvm1;
/*     AliTrackTakenClusterValueMap      atthvm1;  // needed? */
/*     edm::Wrapper<AliTrackTakenClusterValueMap>  watthvm1; // needed? */
  };
}
