#ifndef RecoTracker_MkFitCore_src_MkFinder_h
#define RecoTracker_MkFitCore_src_MkFinder_h

#include "MkBase.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

// Define to get printouts about track and hit chi2.
// See also MkBuilder::backwardFit().

//#define DEBUG_BACKWARD_FIT_BH
//#define DEBUG_BACKWARD_FIT

namespace mkfit {

  class CandCloner;
  class CombCandidate;
  class LayerOfHits;
  class FindingFoos;
  class IterationParams;
  class IterationLayerConfig;
  class SteeringParams;

#if defined(DUMPHITWINDOW) or defined(DEBUG_BACKWARD_FIT)
  class Event;
#endif

  class MkFinder : public MkBase {
    friend class MkBuilder;

  public:
    static constexpr int MPlexHitIdxMax = 16;

    using MPlexHitIdx = Matriplex::Matriplex<int, MPlexHitIdxMax, 1, NN>;
    using MPlexQHoT = Matriplex::Matriplex<HitOnTrack, 1, 1, NN>;

    //----------------------------------------------------------------------------

    MkFinder() {}

    void setup(const PropagationConfig &pc,
               const IterationParams &ip,
               const IterationLayerConfig &ilc,
               const std::vector<bool> *ihm);
    void setup_bkfit(const PropagationConfig &pc);
    void release();

    //----------------------------------------------------------------------------

    void inputTracksAndHitIdx(const std::vector<Track> &tracks, int beg, int end, bool inputProp);

    void inputTracksAndHitIdx(const std::vector<Track> &tracks,
                              const std::vector<int> &idxs,
                              int beg,
                              int end,
                              bool inputProp,
                              int mp_offset);

    void inputTracksAndHitIdx(const std::vector<CombCandidate> &tracks,
                              const std::vector<std::pair<int, int>> &idxs,
                              int beg,
                              int end,
                              bool inputProp);

    void inputTracksAndHitIdx(const std::vector<CombCandidate> &tracks,
                              const std::vector<std::pair<int, IdxChi2List>> &idxs,
                              int beg,
                              int end,
                              bool inputProp);

    void outputTracksAndHitIdx(std::vector<Track> &tracks, int beg, int end, bool outputProp) const;

    void outputTracksAndHitIdx(
        std::vector<Track> &tracks, const std::vector<int> &idxs, int beg, int end, bool outputProp) const;

    void outputTrackAndHitIdx(Track &track, int itrack, bool outputProp) const {
      const int iO = outputProp ? iP : iC;
      copy_out(track, itrack, iO);
    }

    void outputNonStoppedTracksAndHitIdx(
        std::vector<Track> &tracks, const std::vector<int> &idxs, int beg, int end, bool outputProp) const {
      const int iO = outputProp ? iP : iC;
      for (int i = beg, imp = 0; i < end; ++i, ++imp) {
        if (!m_Stopped[imp])
          copy_out(tracks[idxs[i]], imp, iO);
      }
    }

    HitOnTrack bestHitLastHoT(int itrack) const { return m_HoTArrs[itrack][m_NHits(itrack, 0, 0) - 1]; }

    //----------------------------------------------------------------------------

    void getHitSelDynamicWindows(
        const float invpt, const float theta, float &min_dq, float &max_dq, float &min_dphi, float &max_dphi);

    float getHitSelDynamicChi2Cut(const int itrk, const int ipar);

    void selectHitIndices(const LayerOfHits &layer_of_hits, const int N_proc);

    void addBestHit(const LayerOfHits &layer_of_hits, const int N_proc, const FindingFoos &fnd_foos);

    //----------------------------------------------------------------------------

    void findCandidates(const LayerOfHits &layer_of_hits,
                        std::vector<std::vector<TrackCand>> &tmp_candidates,
                        const int offset,
                        const int N_proc,
                        const FindingFoos &fnd_foos);

    //----------------------------------------------------------------------------

    void findCandidatesCloneEngine(const LayerOfHits &layer_of_hits,
                                   CandCloner &cloner,
                                   const int offset,
                                   const int N_proc,
                                   const FindingFoos &fnd_foos);

    void updateWithLastHit(const LayerOfHits &layer_of_hits, int N_proc, const FindingFoos &fnd_foos);

    void copyOutParErr(std::vector<CombCandidate> &seed_cand_vec, int N_proc, bool outputProp) const;

    //----------------------------------------------------------------------------
    // Backward fit

    void bkFitInputTracks(TrackVec &cands, int beg, int end);
    void bkFitOutputTracks(TrackVec &cands, int beg, int end, bool outputProp);

    void bkFitInputTracks(EventOfCombCandidates &eocss, int beg, int end);
    void bkFitOutputTracks(EventOfCombCandidates &eocss, int beg, int end, bool outputProp);

    void bkFitFitTracksBH(const EventOfHits &eventofhits,
                          const SteeringParams &st_par,
                          const int N_proc,
                          bool chiDebug = false);

    void bkFitFitTracks(const EventOfHits &eventofhits,
                        const SteeringParams &st_par,
                        const int N_proc,
                        bool chiDebug = false);

    void bkFitPropTracksToPCA(const int N_proc);

    //----------------------------------------------------------------------------

  private:
    void copy_in(const Track &trk, const int mslot, const int tslot) {
      m_Err[tslot].copyIn(mslot, trk.errors().Array());
      m_Par[tslot].copyIn(mslot, trk.parameters().Array());

      m_Chg(mslot, 0, 0) = trk.charge();
      m_Chi2(mslot, 0, 0) = trk.chi2();
      m_Label(mslot, 0, 0) = trk.label();

      m_NHits(mslot, 0, 0) = trk.nTotalHits();
      m_NFoundHits(mslot, 0, 0) = trk.nFoundHits();

      m_NInsideMinusOneHits(mslot, 0, 0) = trk.nInsideMinusOneHits();
      m_NTailMinusOneHits(mslot, 0, 0) = trk.nTailMinusOneHits();

      std::copy(trk.beginHitsOnTrack(), trk.endHitsOnTrack(), m_HoTArrs[mslot]);
    }

    void copy_out(Track &trk, const int mslot, const int tslot) const {
      m_Err[tslot].copyOut(mslot, trk.errors_nc().Array());
      m_Par[tslot].copyOut(mslot, trk.parameters_nc().Array());

      trk.setCharge(m_Chg(mslot, 0, 0));
      trk.setChi2(m_Chi2(mslot, 0, 0));
      trk.setLabel(m_Label(mslot, 0, 0));

      trk.resizeHits(m_NHits(mslot, 0, 0), m_NFoundHits(mslot, 0, 0));
      std::copy(m_HoTArrs[mslot], &m_HoTArrs[mslot][m_NHits(mslot, 0, 0)], trk.beginHitsOnTrack_nc());
    }

    void copy_in(const TrackCand &trk, const int mslot, const int tslot) {
      m_Err[tslot].copyIn(mslot, trk.errors().Array());
      m_Par[tslot].copyIn(mslot, trk.parameters().Array());

      m_Chg(mslot, 0, 0) = trk.charge();
      m_Chi2(mslot, 0, 0) = trk.chi2();
      m_Label(mslot, 0, 0) = trk.label();

      m_LastHitCcIndex(mslot, 0, 0) = trk.lastCcIndex();
      m_NFoundHits(mslot, 0, 0) = trk.nFoundHits();
      m_NMissingHits(mslot, 0, 0) = trk.nMissingHits();
      m_NOverlapHits(mslot, 0, 0) = trk.nOverlapHits();

      m_NInsideMinusOneHits(mslot, 0, 0) = trk.nInsideMinusOneHits();
      m_NTailMinusOneHits(mslot, 0, 0) = trk.nTailMinusOneHits();

      m_LastHoT[mslot] = trk.getLastHitOnTrack();
      m_CombCand[mslot] = trk.combCandidate();
      m_TrkStatus[mslot] = trk.getStatus();
    }

    void copy_out(TrackCand &trk, const int mslot, const int tslot) const {
      m_Err[tslot].copyOut(mslot, trk.errors_nc().Array());
      m_Par[tslot].copyOut(mslot, trk.parameters_nc().Array());

      trk.setCharge(m_Chg(mslot, 0, 0));
      trk.setChi2(m_Chi2(mslot, 0, 0));
      trk.setLabel(m_Label(mslot, 0, 0));

      trk.setLastCcIndex(m_LastHitCcIndex(mslot, 0, 0));
      trk.setNFoundHits(m_NFoundHits(mslot, 0, 0));
      trk.setNMissingHits(m_NMissingHits(mslot, 0, 0));
      trk.setNOverlapHits(m_NOverlapHits(mslot, 0, 0));

      trk.setNInsideMinusOneHits(m_NInsideMinusOneHits(mslot, 0, 0));
      trk.setNTailMinusOneHits(m_NTailMinusOneHits(mslot, 0, 0));

      trk.setCombCandidate(m_CombCand[mslot]);
      trk.setStatus(m_TrkStatus[mslot]);
    }

    void add_hit(const int mslot, int index, int layer) {
      // Only used by BestHit.
      // m_NInsideMinusOneHits and m_NTailMinusOneHits are maintained here but are
      // not used and are not copied out (as Track does not have these members).

      int &n_tot_hits = m_NHits(mslot, 0, 0);
      int &n_fnd_hits = m_NFoundHits(mslot, 0, 0);

      if (n_tot_hits < Config::nMaxTrkHits) {
        m_HoTArrs[mslot][n_tot_hits++] = {index, layer};
        if (index >= 0) {
          ++n_fnd_hits;
          m_NInsideMinusOneHits(mslot, 0, 0) += m_NTailMinusOneHits(mslot, 0, 0);
          m_NTailMinusOneHits(mslot, 0, 0) = 0;
        } else if (index == -1) {
          ++m_NTailMinusOneHits(mslot, 0, 0);
        }
      } else {
        // printf("WARNING MkFinder::add_hit hit-on-track limit reached for label=%d\n", label_);

        const int LH = Config::nMaxTrkHits - 1;

        if (index >= 0) {
          if (m_HoTArrs[mslot][LH].index < 0)
            ++n_fnd_hits;
          m_HoTArrs[mslot][LH] = {index, layer};
        } else if (index == -2) {
          if (m_HoTArrs[mslot][LH].index >= 0)
            --n_fnd_hits;
          m_HoTArrs[mslot][LH] = {index, layer};
        }
      }
    }

    int num_all_minus_one_hits(const int mslot) const {
      return m_NInsideMinusOneHits(mslot, 0, 0) + m_NTailMinusOneHits(mslot, 0, 0);
    }

    int num_inside_minus_one_hits(const int mslot) const { return m_NInsideMinusOneHits(mslot, 0, 0); }

    //----------------------------------------------------------------------------

    MPlexQF m_Chi2;
    MPlexQI m_Label;  // seed index in global seed vector (for MC truth match)

    MPlexQI m_NHits;
    MPlexQI m_NFoundHits;

    HitOnTrack m_HoTArrs[NN][Config::nMaxTrkHits];

#if defined(DUMPHITWINDOW) or defined(DEBUG_BACKWARD_FIT)
    MPlexQI m_SeedAlgo;   // seed algorithm
    MPlexQI m_SeedLabel;  // seed label
    Event *m_event;
#endif

    MPlexQI m_SeedIdx;  // seed index in local thread (for bookkeeping at thread level)
    MPlexQI m_CandIdx;  // candidate index for the given seed (for bookkeeping of clone engine)

    MPlexQI m_Stopped;  // Flag for BestHit that a track has been stopped (and copied out already)

    // Additions / substitutions for TrackCand copy_in/out()
    // One could really access the original TrackCand for all of those, especially the ones that
    // are STD only. This then requires access back to that TrackCand memory.
    // So maybe one should just have flags for CopyIn methods (or several versions). Yay, etc.
    MPlexQI m_NMissingHits;             // sub: m_NHits, sort of, STD only
    MPlexQI m_NOverlapHits;             // add: num of overlaps registered in HitOnTrack, STD only
    MPlexQI m_NInsideMinusOneHits;      // sub: before we copied all hit idcs and had a loop counting them only
    MPlexQI m_NTailMinusOneHits;        // sub: before we copied all hit idcs and had a loop counting them only
    MPlexQI m_LastHitCcIndex;           // add: index of last hit in m_CombCand hit tree, STD only
    TrackBase::Status m_TrkStatus[NN];  // STD only, status bits
    HitOnTrack m_LastHoT[NN];
    CombCandidate *m_CombCand[NN];
    // const TrackCand *m_TrkCand[NN]; // hmmh, could get all data through this guy ... but scattered
    // storing it in now for bkfit debug printouts
    TrackCand *m_TrkCand[NN];

    // Hit indices into LayerOfHits to explore.
    WSR_Result m_XWsrResult[NN];  // Could also merge it with m_XHitSize. Or use smaller arrays.
    MPlexQI m_XHitSize;
    MPlexHitIdx m_XHitArr;

    // Hit errors / parameters for hit matching, update.
    MPlexHS m_msErr;
    MPlexHV m_msPar;

    // An idea: Do propagation to hit in FindTracksXYZZ functions.
    // Have some state / functions here that make this short to write.
    // This would simplify KalmanUtils (remove the propagate functions).
    // Track errors / parameters propagated to current hit.
    // MPlexLS    candErrAtCurrHit;
    // MPlexLV    candParAtCurrHit;

    const PropagationConfig *m_prop_config = nullptr;
    const IterationParams *m_iteration_params = nullptr;
    const IterationLayerConfig *m_iteration_layer_config = nullptr;
    const std::vector<bool> *m_iteration_hit_mask = nullptr;

    // Backward fit
    int m_CurHit[NN];
    const HitOnTrack *m_HoTArr[NN];
    int m_CurNode[NN];
    const HoTNode *m_HoTNodeArr[NN];
  };

}  // end namespace mkfit
#endif
