#ifndef RecoTracker_MkFitCore_src_MkFitter_h
#define RecoTracker_MkFitCore_src_MkFitter_h

#include "MkBase.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/HitStateOnTrack.h"

#include <vector>

namespace mkfit {

  class CandCloner;
  class Event;

  static constexpr int MPlexHitIdxMax = 16;
  using MPlexHitIdx = Matriplex::Matriplex<int, MPlexHitIdxMax, 1, NN>;
  using MPlexQHoT = Matriplex::Matriplex<HitOnTrack, 1, 1, NN>;

  class MkFitter : public MkBase {
    friend class MkBuilder;

  public:
    MkFitter() {}

    //----------------------------------------------------------------------------

    void fwdFitInputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end);
    void fwdFitFitTracks(const EventOfHits &eventofhits,
                         const int N_proc,
                         int nFoundHits,
                         std::vector<std::vector<int>> indices_R2Z,
                         float *chi2);
    void bkReFitInputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end, int nFoundHits);
    void bkReFitFitTracks(const EventOfHits &eventofhits,
                          const int N_proc,
                          int nFoundHits,
                          std::vector<std::vector<int>> indices_R2Z,
                          float *chi2);
    void reFitOutputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end, int nFoundHits, bool bkw = false);
    void storeHitStates(const int h,
                        const int nFoundHits,
                        const int N_proc,
                        const int *hot,
                        const MPlex5V &bkPredPar,
                        const MPlex5S &bkPredErr,
                        const MPlex5V &bkUpdPar,
                        const MPlex5S &bkUpdErr,
                        const MPlexQI &bkPzSign,
                        const MPlexQF &bkChi2);
    std::vector<std::vector<int>> reFitIndices(const EventOfHits &eventofhits, const int N_proc, int nFoundHits);

    void set_cpe(cpe_func cpe_function) { m_cpe_corr_func = cpe_function; };

    //----------------------------------------------------------------------------

    void release();

  private:
    MPlexQF m_Chi2;

    // Hit errors / parameters for update.
    MPlexHS m_msErr{0.0f};
    MPlexHV m_msPar{0.0f};

    int m_CurHit[NN];
    const HitOnTrack *m_HoTArr[NN];

    const Event *m_event = nullptr;
    const PropagationFlags *refit_flags = nullptr;
    const FinalFitFlags *refit_ffflags = nullptr;
    cpe_func m_cpe_corr_func = nullptr;

    // Per-hit states of the final fit (MkBuilder::set_hit_states_output).  m_hsOut[i]: the states of the
    // track in lane i (nullptr = not stored); m_hsFwdOut / m_hsBwdOut (validation only): the forward updated and
    // backward predicted states the smoother combines.  m_fwdLoc*: forward updated state at the h-th hit of the
    // forward pass in the module's local frame (as the update computes it), kept for the backward pass;
    // m_fwdChi2Outer: the forward chi2 of the outermost hit.
    bool m_storeHitStates = false;
    bool m_validateHitStates = false;
    bool m_storeOuter = false;  // the state at the outermost hit, m_outerOut[i] (also without per-hit states)
    HitStateOnTrack *m_hsOut[NN];
    OuterHitState *m_outerOut[NN];
    HitStateOnTrack *m_hsFwdOut[NN];
    HitStateOnTrack *m_hsBwdOut[NN];
    std::vector<MPlex5V> m_fwdLocPar;
    std::vector<MPlex5S> m_fwdLocErr;
    std::vector<MPlexQI> m_fwdPzSign;
    MPlexQF m_fwdChi2Outer;
  };

}  // end namespace mkfit

#endif
