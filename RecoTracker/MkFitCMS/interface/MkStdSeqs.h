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

  namespace StdSeq {

    void loadDeads(EventOfHits &eoh, const std::vector<DeadVec> &deadvectors);

    void cmssw_LoadHits_Begin(EventOfHits &eoh, const std::vector<const HitVec *> &orig_hitvectors);
    void cmssw_LoadHits_End(EventOfHits &eoh);

    // Not used anymore. Left here if we want to experiment again with
    // COPY_SORTED_HITS in class LayerOfHits.
    void cmssw_Map_TrackHitIndices(const EventOfHits &eoh, TrackVec &seeds);
    void cmssw_ReMap_TrackHitIndices(const EventOfHits &eoh, TrackVec &out_tracks);

    int clean_cms_seedtracks_iter(TrackVec *seed_ptr, const IterationConfig &itrcfg, const BeamSpot &bspot);

    void find_duplicates(TrackVec &tracks);
    void remove_duplicates(TrackVec &tracks);

    void find_duplicates_sharedhits(TrackVec &tracks, const float fraction);
    void find_duplicates_sharedhits_pixelseed(TrackVec &tracks,
                                              const float fraction,
                                              const float drth_central,
                                              const float drth_obarrel,
                                              const float drth_forward);

    // quality filter for n hits with seed hit "penalty" for strip-based seeds
    //   this implicitly separates triplets and doublet seeds with glued layers
    template <class TRACK>
    bool qfilter_n_hits(const TRACK &t, int nMinHits) {
      int seedHits = t.getNSeedHits();
      int seedReduction = (seedHits <= 5) ? 2 : 3;
      return t.nFoundHits() - seedReduction >= nMinHits;
    }

    // simple hit-count quality filter; used with pixel-based seeds
    template <class TRACK>
    bool qfilter_n_hits_pixseed(const TRACK &t, int nMinHits) {
      return t.nFoundHits() >= nMinHits;
    }

    // layer-dependent quality filter
    template <class TRACK>
    bool qfilter_n_layers(const TRACK &t, const BeamSpot &bspot, const TrackerInfo &trk_inf) {
      int enhits = t.nHitsByTypeEncoded(trk_inf);
      int npixhits = t.nPixelDecoded(enhits);
      int enlyrs = t.nLayersByTypeEncoded(trk_inf);
      int npixlyrs = t.nPixelDecoded(enlyrs);
      int nmatlyrs = t.nTotMatchDecoded(enlyrs);
      int llyr = t.getLastFoundHitLyr();
      int lplyr = t.getLastFoundPixelHitLyr();
      float invpt = t.invpT();
      float invptmin = 1.43;  // min 1/pT (=1/0.7) for full filter on (npixhits<=3 .or. npixlyrs<=3)
      float d0BS = t.d0BeamSpot(bspot.x, bspot.y);
      float d0_max = 0.1;  // 1 mm

      bool endsInsidePix = (llyr == 2 || llyr == 18 || llyr == 45);
      bool lastInsidePix = ((0 <= lplyr && lplyr < 3) || (18 <= lplyr && lplyr < 20) || (45 <= lplyr && lplyr < 47));
      return !(((npixhits <= 3 || npixlyrs <= 3) && endsInsidePix &&
                (invpt < invptmin || (invpt >= invptmin && std::abs(d0BS) > d0_max))) ||
               ((npixlyrs <= 3 && nmatlyrs <= 6) && lastInsidePix && llyr != lplyr && std::abs(d0BS) > d0_max));
    }

    /// quality filter tuned for pixelLess iteration during forward search
    template <class TRACK>
    bool qfilter_pixelLessFwd(const TRACK &t, const BeamSpot &bspot, const TrackerInfo &tk_info) {
      float d0BS = t.d0BeamSpot(bspot.x, bspot.y);
      float d0_max = 0.05;  // 0.5 mm

      int encoded;
      encoded = t.nLayersByTypeEncoded(tk_info);
      int nLyrs = t.nTotMatchDecoded(encoded);
      encoded = t.nHitsByTypeEncoded(tk_info);
      int nHits = t.nTotMatchDecoded(encoded);

      int seedReduction = (t.getNSeedHits() <= 5) ? 2 : 3;

      float invpt = t.invpT();
      float invptmin = 1.11;  // =1/0.9

      float thetasym = std::abs(t.theta() - Const::PIOver2);
      float thetasymmin = 1.11;  // -> |eta|=1.45

      return (((t.nFoundHits() - seedReduction >= 4 && invpt < invptmin) ||
               (t.nFoundHits() - seedReduction >= 3 && invpt > invptmin && thetasym <= thetasymmin) ||
               (t.nFoundHits() - seedReduction >= 4 && invpt > invptmin && thetasym > thetasymmin)) &&
              !((nLyrs <= 4 || nHits <= 4) && std::abs(d0BS) > d0_max && invpt < invptmin));
    }

    /// quality filter tuned for pixelLess iteration during backward search
    template <class TRACK>
    bool qfilter_pixelLessBkwd(const TRACK &t, const BeamSpot &bspot, const TrackerInfo &tk_info) {
      float d0BS = t.d0BeamSpot(bspot.x, bspot.y);
      float d0_max = 0.1;  // 1 mm

      int encoded;
      encoded = t.nLayersByTypeEncoded(tk_info);
      int nLyrs = t.nTotMatchDecoded(encoded);
      encoded = t.nHitsByTypeEncoded(tk_info);
      int nHits = t.nTotMatchDecoded(encoded);

      float invpt = t.invpT();
      float invptmin = 1.11;  // =1/0.9

      float thetasym = std::abs(t.theta() - Const::PIOver2);
      float thetasymmin_l = 0.80;  // -> |eta|=0.9
      float thetasymmin_h = 1.11;  // -> |eta|=1.45

      return !(
          ((nLyrs <= 3 || nHits <= 3)) ||
          ((nLyrs <= 4 || nHits <= 4) && (invpt < invptmin || (thetasym > thetasymmin_l && std::abs(d0BS) > d0_max))) ||
          ((nLyrs <= 6 || nHits <= 6) && (invpt > invptmin && thetasym > thetasymmin_h && std::abs(d0BS) > d0_max)));
    }

    template <class TRACK>
    bool qfilter_nan_n_silly(const TRACK &t) {
      return !(t.hasNanNSillyValues());
    }

    void find_and_remove_duplicates(TrackVec &tracks, const IterationConfig &itconf);

  }  // namespace StdSeq

}  // namespace mkfit

#endif
