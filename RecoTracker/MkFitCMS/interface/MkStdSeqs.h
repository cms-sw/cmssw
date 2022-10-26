#ifndef RecoTracker_MkFitCMS_interface_MkStdSeqs_h
#define RecoTracker_MkFitCMS_interface_MkStdSeqs_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

namespace mkfit {

  class EventOfHits;
  class IterationConfig;
  class TrackerInfo;
  class MkJob;
  class TrackCand;

  namespace StdSeq {

    void loadDeads(EventOfHits &eoh, const std::vector<DeadVec> &deadvectors);

    void cmssw_LoadHits_Begin(EventOfHits &eoh, const std::vector<const HitVec *> &orig_hitvectors);
    void cmssw_LoadHits_End(EventOfHits &eoh);

    // Not used anymore. Left here if we want to experiment again with
    // COPY_SORTED_HITS in class LayerOfHits.
    void cmssw_Map_TrackHitIndices(const EventOfHits &eoh, TrackVec &seeds);
    void cmssw_ReMap_TrackHitIndices(const EventOfHits &eoh, TrackVec &out_tracks);

    int clean_cms_seedtracks_iter(TrackVec &seeds, const IterationConfig &itrcfg, const BeamSpot &bspot);

    void remove_duplicates(TrackVec &tracks);

    void clean_duplicates(TrackVec &tracks, const IterationConfig &itconf);
    void clean_duplicates_sharedhits(TrackVec &tracks, const IterationConfig &itconf);
    void clean_duplicates_sharedhits_pixelseed(TrackVec &tracks, const IterationConfig &itconf);

    // Quality filters used directly (not through IterationConfig)

    template <class TRACK>
    bool qfilter_nan_n_silly(const TRACK &t, const MkJob &) {
      return !(t.hasNanNSillyValues());
    }

  }  // namespace StdSeq

}  // namespace mkfit

#endif
