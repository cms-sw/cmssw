#ifndef RecoTracker_MkFitCore_standalone_RntDumper_RntConversions_h
#define RecoTracker_MkFitCore_standalone_RntDumper_RRntConversions_h

#include "RecoTracker/MkFitCore/standalone/RntDumper/RntStructs.h"

#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/src/Matrix.h"
#include "RecoTracker/MkFitCore/src/MiniPropagators.h"

namespace mkfit {
  namespace miprops = mkfit::mini_propagators;

  RVec state2pos(const miprops::State &s) { return {s.x, s.y, s.z}; }
  RVec state2mom(const miprops::State &s) { return {s.px, s.py, s.pz}; }
  State state2state(const miprops::State &s) { return {state2pos(s), state2mom(s)}; }

  RVec statep2pos(const miprops::StatePlex &s, int i) { return {s.x[i], s.y[i], s.z[i]}; }
  RVec statep2mom(const miprops::StatePlex &s, int i) { return {s.px[i], s.py[i], s.pz[i]}; }
  State statep2state(const miprops::StatePlex &s, int i) { return {statep2pos(s, i), statep2mom(s, i)}; }
  PropState statep2propstate(const miprops::StatePlex &s, int i) {
    return {statep2state(s, i), s.dalpha[i], s.fail_flag[i]};
  }

  RVec hit2pos(const Hit &h) { return {h.x(), h.y(), h.z()}; }
  RVec track2pos(const TrackBase &s) { return {s.x(), s.y(), s.z()}; }
  RVec track2mom(const TrackBase &s) { return {s.px(), s.py(), s.pz()}; }
  State track2state(const TrackBase &s) { return {track2pos(s), track2mom(s)}; }

  SimSeedInfo evsi2ssinfo(const Event *ev, int seed_idx) {
    SimSeedInfo ssi;
    Event::SimLabelFromHits slfh = ev->simLabelForCurrentSeed(seed_idx);
    if (slfh.is_set()) {
      ssi.s_sim = track2state(ev->simTracks_[slfh.label]);
      ssi.sim_lbl = slfh.label;
      ssi.n_hits = slfh.n_hits;
      ssi.n_match = slfh.n_match;
      ssi.has_sim = true;
    }
    auto seed = ev->currentSeed(seed_idx);
    ssi.s_seed = track2state(seed);
    ssi.seed_lbl = seed.label();
    ssi.seed_idx = seed_idx;
    return ssi;
  }
}  // namespace mkfit

#endif
