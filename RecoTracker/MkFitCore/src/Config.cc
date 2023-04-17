#include "RecoTracker/MkFitCore/interface/Config.h"

namespace mkfit {

  namespace Config {
    // Multi threading configuration
#if defined(MKFIT_STANDALONE)
    int numThreadsFinder = 1;
    int numThreadsEvents = 1;
    int numSeedsPerTask = 32;
#endif

#if defined(MKFIT_STANDALONE)
    bool removeDuplicates = false;
    bool useHitsForDuplicates = true;
#endif
    const float maxdPt = 0.5;
    const float maxdPhi = 0.25;
    const float maxdEta = 0.05;
    const float maxdR = 0.0025;
    const float minFracHitsShared = 0.75;

    const float maxd1pt = 1.8;     //windows for hit
    const float maxdphi = 0.37;    //and/or dr
    const float maxdcth = 0.37;    //comparisons
    const float maxcth_ob = 1.99;  //eta 1.44
    const float maxcth_fw = 6.05;  //eta 2.5

#ifdef CONFIG_PhiQArrays
    bool usePhiQArrays = true;
#endif
  }  // namespace Config

}  // end namespace mkfit
