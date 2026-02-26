#ifndef RecoTracker_MkFitCore_src_MkFitter_h
#define RecoTracker_MkFitCore_src_MkFitter_h

#include "MkBase.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

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
    void bkReFitInputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end);
    void bkReFitFitTracks(const EventOfHits &eventofhits,
                          const int N_proc,
                          int nFoundHits,
                          std::vector<std::vector<int>> indices_R2Z,
                          float *chi2);
    void reFitOutputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end, int nFoundHits, bool bkw = false);
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
    cpe_func m_cpe_corr_func = nullptr;
  };

}  // end namespace mkfit

#endif
