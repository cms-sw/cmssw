#ifndef RecoTracker_MkFitCMS_interface_buildtestMPlex_h
#define RecoTracker_MkFitCMS_interface_buildtestMPlex_h

#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"

#include <sys/time.h>

namespace mkfit {

  class IterationConfig;
  class MkBuilder;

  void runBuildingTestPlexDumbCMSSW(Event& ev, const EventOfHits& eoh, MkBuilder& builder);

  double runBuildingTestPlexBestHit(Event& ev, const EventOfHits& eoh, MkBuilder& builder);
  double runBuildingTestPlexStandard(Event& ev, const EventOfHits& eoh, MkBuilder& builder);
  double runBuildingTestPlexCloneEngine(Event& ev, const EventOfHits& eoh, MkBuilder& builder);

  std::vector<double> runBtpCe_MultiIter(Event& ev, const EventOfHits& eoh, MkBuilder& builder, int n);

  inline double dtime() {
    double tseconds = 0.0;
    struct timeval mytime;
    gettimeofday(&mytime, (struct timezone*)nullptr);
    tseconds = (double)(mytime.tv_sec + mytime.tv_usec * 1.0e-6);
    return (tseconds);
  }

}  // end namespace mkfit
#endif
