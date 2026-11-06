#ifndef RecoTracker_MkFitCore_interface_HitStateOnTrack_h
#define RecoTracker_MkFitCore_interface_HitStateOnTrack_h

#include <span>
#include <vector>

namespace mkfit {

  // State of the final fit on the module plane of one hit, in that module's local frame as
  // LocalTrajectoryParameters / LocalTrajectoryError: par = (q/p, dx/dz, dy/dz, x, y), err = lower triangle of the
  // 5x5 covariance (00, 10, 11, 20, 21, 22, ...), pzSign = sign of the local z momentum.  A track has one entry per
  // HitOnTrack position; valid is false where the final fit did not use that position (missing or removed hit).
  struct HitStateOnTrack {
    enum Kind : signed char { Combined = 0, ForwardOnly = 1, BackwardOnly = 2 };

    float par[5];
    float err[15];
    float chi2;  // the hit's chi2 increment in the final fit
    signed char pzSign;
    signed char kind;  // how the smoothed state was obtained (Kind)
    bool valid = false;
  };

  // The HitStateOnTrack entries of a set of tracks in one array: track i has one entry per HitOnTrack position, from
  // m_offsets[i] to m_offsets[i + 1].  operator[](i) gives the entries of track i.  Empty: no tracks, or not stored.
  class HitStatesOnTracks {
  public:
    // Entries for n tracks, nHits(i) for track i, all invalid.
    template <typename F>
    void reset(int n, F&& nHits) {
      m_offsets.resize(n + 1);
      m_offsets[0] = 0;
      for (int i = 0; i < n; ++i)
        m_offsets[i + 1] = m_offsets[i] + nHits(i);
      m_states.assign(m_offsets[n], HitStateOnTrack{});
    }

    int size() const { return m_offsets.empty() ? 0 : int(m_offsets.size()) - 1; }
    bool empty() const { return size() == 0; }

    std::span<HitStateOnTrack> operator[](int i) {
      return {m_states.data() + m_offsets[i], m_states.data() + m_offsets[i + 1]};
    }
    std::span<const HitStateOnTrack> operator[](int i) const {
      return {m_states.data() + m_offsets[i], m_states.data() + m_offsets[i + 1]};
    }

  private:
    std::vector<HitStateOnTrack> m_states;
    std::vector<int> m_offsets;
  };

  // The state at the outermost hit the final fit used, and that hit's HitOnTrack position (-1: none): the smoothed
  // state there, as the per-hit states have it.
  struct OuterHitState {
    HitStateOnTrack state;
    int hot = -1;
  };

  // The outputs of the final fit besides the tracks: the state at the outermost hit of every track (always), and with
  // perHit the smoothed state at every hit; with validate, also the forward updated and the backward predicted state
  // each smoothed state combines.
  struct FinalFitHitStates {
    std::vector<OuterHitState> outer;
    HitStatesOnTracks smoothed;
    HitStatesOnTracks fwd;
    HitStatesOnTracks bwd;
    bool perHit = true;
    bool validate = false;

    template <typename F>
    void reset(int n, F&& nHits) {
      outer.assign(n, OuterHitState{});
      if (perHit)
        smoothed.reset(n, nHits);
      if (perHit && validate) {
        fwd.reset(n, nHits);
        bwd.reset(n, nHits);
      }
    }
  };

}  // namespace mkfit

#endif
