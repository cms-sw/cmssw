#ifndef RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h
#define RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include "MiniPropagators.h"
#include "V2p2Score.h"

#include "V2p2Config.h"
#include <algorithm>
#include <cassert>
#include <new>
#include <type_traits>
#include <vector>

namespace mkfit {

#ifdef MKFIT_TRACE
  class Event;
#endif

  struct CCandRep;
  struct CandIncoming;

  //----------------------------------------------------------------------------

  // A vector with fixed inline capacity and no construction of the unused
  // slots, for trivially copyable T. Per-candidate, per-layer storage that
  // std::vector would allocate and free on every layer.
  template <typename T, int Cap>
  class InlineVec {
    static_assert(std::is_trivially_copyable_v<T>);
    alignas(T) unsigned char m_buf[Cap * sizeof(T)];
    int m_size = 0;

  public:
    static constexpr int capacity() { return Cap; }
    int size() const { return m_size; }
    bool empty() const { return m_size == 0; }
    void clear() { m_size = 0; }
    T* begin() { return std::launder(reinterpret_cast<T*>(m_buf)); }
    T* end() { return begin() + m_size; }
    const T* begin() const { return std::launder(reinterpret_cast<const T*>(m_buf)); }
    const T* end() const { return begin() + m_size; }
    T& operator[](int i) { return begin()[i]; }
    const T& operator[](int i) const { return begin()[i]; }
    T& front() { return begin()[0]; }
    void push_back(const T& v) {
      assert(m_size < Cap);
      new (m_buf + m_size * sizeof(T)) T(v);
      ++m_size;
    }
    void pop_back() { --m_size; }
  };

  //----------------------------------------------------------------------------

  // One node of the in-layer combinatorial tree: a candidate that has taken a
  // particular sequence of hits within this layer, with the Kalman updates
  // applied. Nodes live in MkFinderV2p2::m_node_arena, are reached by index and
  // link only to their parent. See doc/MkFinderV2p2-DesignNotes.md,
  // "In-layer combinatorial search".
  struct CandNode {
    CandIncoming* m_cand;  // the candidate this path extends
    int m_parent_idx;      // arena index of the predecessor; -1 = the CandIncoming itself
    int m_hit_pos;         // position of this hit in m_cand->m_layer_hits; the next
                           // hit a child may take is at m_hit_pos + 1
    HitOnTrack m_hot;
    TrackState m_state;  // AFTER the update at m_hot
    float m_chi2;        // of this hit alone
    // The whole path through this layer, accumulated parent -> child: one layer
    // step however many hits the path took.
    LayerStepFeatures m_feat;
#ifdef MKFIT_TRACE
    int m_tr_hitmatch_id = -1;
#endif
  };

  //----------------------------------------------------------------------------

  // One TrackCand of a CombCandidate entering the layer. Its state stays in the
  // TrackCand (tcand()); this carries the layer's work for it: the pre-selection
  // queues per sub-layer, the path-ordered hits, the WSR verdict and the local
  // hit density. Lives in CCandRep::m_cands for one layer.
  struct CandIncoming {
    CCandRep* mp_ccrep;
    int m_origin_tcand_index;  // TrackCand index in CombCandidate

    // A hit that passed pre-selection for this candidate: a hit compatible with
    // it, not a truth match. The trace records it as a TrHitMatch with
    // passed_pqueue set. Held in m_pqueue, then in m_layer_hits.
    struct PassedHitMatch {
      float ddphi;  // the pre-selection's ranking: phi distance to the hit, lower is better
      unsigned int hit_orig_index;
      unsigned int hit_index;
      int layer;

      // Position and momentum at the crossing with the hit's module plane (the
      // Hermite cubic solved onto it), with inv_pt, inv_k and theta of the state at
      // the previous hit. Its dalpha orders the hits along the path.
      mini_propagators::InitialState mixed_state;
#ifdef MKFIT_TRACE
      int tr_hitmatch_id = -1;
#endif

      // Worst (largest ddphi) on top, so it is the one replaced.
      bool operator<(const PassedHitMatch& o) const { return ddphi < o.ddphi; }
    };
    // One bounded pqueue per sub-layer, [0] primary and [1] secondary, so each
    // sensor has its own reduction budget. See doc/MkFinderV2p2-DesignNotes.md,
    // "Reduction and hit ordering".
    // Max-heaps on ddphi, worst on top, run with std::push_heap / pop_heap
    // exactly as std::priority_queue<PassedHitMatch> does, in inline storage.
    static constexpr int kPqCap = Config::V2p2::InLayer::max_presel_hits_limit;
    InlineVec<PassedHitMatch, kPqCap> m_pqueue[2];

    void pq_push(int sl, const PassedHitMatch& e) {
      m_pqueue[sl].push_back(e);
      std::push_heap(m_pqueue[sl].begin(), m_pqueue[sl].end());
    }
    const PassedHitMatch& pq_top(int sl) { return m_pqueue[sl].front(); }
    void pq_pop(int sl) {
      std::pop_heap(m_pqueue[sl].begin(), m_pqueue[sl].end());
      m_pqueue[sl].pop_back();
    }

    // Survivors of both queues, in path order (sorted by dir * dalpha).
    InlineVec<PassedHitMatch, 2 * kPqCap> m_layer_hits;

    float m_log_rho = 0.0f;  // local hit density, ln(hits / cm^2), for the score

    CandIncoming(CCandRep* ccr, int orig_idx) {
      mp_ccrep = ccr;
      m_origin_tcand_index = orig_idx;
    }

    // Within-sensitive-region verdict for this candidate on this layer, set by
    // MkFinderV2p2::determine_wsr():
    //   WSR_Inside  -- the crossing is inside sensitive q; a missing hit is a hole;
    //   WSR_Edge    -- the crossing is near a boundary, or the track turns round
    //                  inside the layer; a missing hit is not counted;
    //   WSR_Outside -- the crossing misses the layer's sensitive extent, or lies
    //                  wholly in a disc's central hole; the layer is skipped.
    WSR_Result m_wsr;

    CombCandidate& ccand();
    TrackCand& tcand();
    const TrackCand& tcand() const;
  };  // end struct CandIncoming

  //----------------------------------------------------------------------------

  // One CombCandidate (one seed) in the search: its incoming candidates in this layer,
  // the arena indices of all their in-layer paths, and the counts that say when
  // it can be selected. Lives in MkFinderV2p2::m_active_ccreps from pickup until
  // all its TrackCands have stopped.
  struct CCandRep {
    CombCandidate& m_ccand;

    std::vector<CandIncoming> m_cands;

    // Arena indices of every in-layer path of this CombCandidate, across all its
    // incoming candidates, for the end-of-layer selection.
    std::vector<int> m_nodes;

    // Incoming candidates of this layer still in MkFinderV2p2::m_cand_queue. They are
    // queued together, so at zero all of them have been through pre-selection.
    int m_n_queued = 0;
    // Kalman items in flight plus nodes waiting to be expanded. Selection needs
    // both counts at zero.
    int m_n_inflight = 0;
    bool m_selected = false;  // select_and_materialise() already ran this layer
    // Beam width this layer, set at pull-in in begin_next_Ccrep_in_layer() (0
    // before the first pull-in: the CombCandidate's capacity). See
    // InLayer::late_max_cands.
    int m_max_cands = 0;

    // m_cand_queue holds pointers into m_cands, so it must never reallocate:
    // a CombCandidate never has more TrackCands than its capacity.
    CCandRep(CombCandidate& ccand) : m_ccand(ccand) { m_cands.reserve(ccand.capacity()); }
  };  // end struct CCandRep

  inline CombCandidate& CandIncoming::ccand() { return mp_ccrep->m_ccand; }
  inline TrackCand& CandIncoming::tcand() { return mp_ccrep->m_ccand[m_origin_tcand_index]; }
  inline const TrackCand& CandIncoming::tcand() const { return mp_ccrep->m_ccand[m_origin_tcand_index]; }

  //----------------------------------------------------------------------------

  // The track side of a Kalman batch, NN lanes: the state at the previous hit,
  // the propagated state and the path length to the module plane.
  struct BaseArgs {
    const PropagationConfig* prop_config = nullptr;

    mini_propagators::InitialStatePlex tsXyz;
    MPlexLS tsErr{0.0f};  // input (on prev hit) and output (on current hit) [ts - track-state]
    MPlexLV tsPar{0.0f};  // ""
    MPlexQI tsChg{0};     // "" Kalman update can flip it through curvature flip

    MPlexQF sPerp{0.0f};  // transverse path length, from dalpha in compute_pars(); path_from_perp() makes it 3D

    MPlexLS propErr{0.0f};  // intermediate: propagated from tsErr and used as input to Kalman
    MPlexLV propPar{0.0f};  // sPerp path: input, the search's crossing (compute_pars()); solve-plane path: output

    MPlexQI outFailFlag{0};  // not checked: the crossing is known from the search, or a few cm away within the layer

    int N_filled = 0;
  };

  //----------------------------------------------------------------------------

  // One Kalman batch of the in-layer search, NN (candidate, hit) items: the hits
  // and module planes, where each lane came from, and the outcomes, appended to
  // *mp_out by do_kalman_stuff().
  struct KalmanOpArgs : public BaseArgs {
    CandIncoming* candp[NN];
    HitOnTrack hot[NN];
    // Provenance of each lane, for the in-layer expansion: which arena node this
    // extension grew from (-1 = the CandIncoming's own incoming state) and which
    // entry of m_layer_hits it takes.
    int parent_idx[NN];
    int hit_pos[NN];

    MPlexHS msErr{0.0f};  // input measurement / hit [ms - measurement state]
    MPlexHV msPar{0.0f};  // ""
    MPlexHV plNrm{0.0f};  // input detector plane [pl - plane]
    MPlexHV plDir{0.0f};  // ""
    MPlexHV plPnt{0.0f};  // ""

    MPlexQF tsChi2{0.0f};     // output
    MPlexQF tsDetV{0.0f};     // output: det of the 2x2 residual covariance
    MPlexQF tsChi2Acc{0.0f};  // output: acceptance chi2, V2p2Params::chi2_trk_fac

#ifdef MKFIT_TRACE
    int tr_hitmatch_ids[NN];
    void set_tr_hitmatch_id(int id) { tr_hitmatch_ids[N_filled] = id; }
    const Event* mp_event = nullptr;
#endif

    // One result per lane, appended to *mp_out.
    struct ItemOut {
      CandIncoming* cand;
      int parent_idx;
      int hit_pos;
      HitOnTrack hot;
      float chi2;
      float det_v;     // det of the 2x2 residual covariance, for the score
      float chi2_acc;  // acceptance chi2, compared with V2p2Params::hit_chi2_cut
      TrackState state;
#ifdef MKFIT_TRACE
      int tr_hitmatch_id;
#endif
    };
    std::vector<ItemOut>* mp_out = nullptr;

    // Solve for the crossing with the module plane instead of taking sPerp and
    // propPar from the search. Set for depth >= 1 of the in-layer search, where
    // the Hermite crossings no longer apply.
    bool m_solve_plane = false;
    // Scale on the track sigma in the acceptance chi2, V2p2Params::chi2_trk_fac.
    float m_chi2_trk_fac = 1.0f;

    void reset() { N_filled = 0; }

    void item_begin(CandIncoming* cand, HitOnTrack ht, int par_idx = -1, int hpos = -1) {
      candp[N_filled] = cand;
      hot[N_filled] = ht;
      parent_idx[N_filled] = par_idx;
      hit_pos[N_filled] = hpos;
    }
    bool item_finished() { return ++N_filled == NN; }

    // Start from a TrackState directly, for the solve-plane path: there is no
    // pre-solved crossing, so tsXyz is not filled and compute_pars() must be
    // skipped -- propPar is a pure output there, where on the sPerp path it is
    // an input carrying the already-known intersection.
    void load_state_err_chg(const TrackState& st) {
      tsPar.copyIn(N_filled, st.parArray());
      tsErr.copyIn(N_filled, st.errArray());
      tsChg[N_filled] = st.charge;
    }

    void load_state_err_chg(const mini_propagators::InitialState& params_on_hit, const TrackState& prev_state) {
      tsXyz.copyIn(N_filled, params_on_hit);
      tsPar.copyIn(N_filled, prev_state.parArray());  // the propagation steps need the initial parameters, too
      tsErr.copyIn(N_filled, prev_state.errArray());
      tsChg[N_filled] = prev_state.charge;
    }

    void load_hit_module(const Hit& hit, const ModuleInfo& mi) {
      msErr.copyIn(N_filled, hit.errArray());
      msPar.copyIn(N_filled, hit.posArray());
      plNrm.copyIn(N_filled, mi.zdir.Array());
      plDir.copyIn(N_filled, mi.xdir.Array());
      plPnt.copyIn(N_filled, mi.pos.Array());
    }

    void compute_pars() {
      // Computed at full width: the free lanes take lane 0, as pad_free_lanes() does for the
      // other inputs in do_kalman_stuff(), so that no lane works on a previous batch's values
      // or on memory never loaded.
      for (int i = N_filled; i < NN; ++i)
        tsXyz.copyIn(i, tsXyz, 0);
      // The search's crossing on the module plane (tsXyz) becomes propPar and the
      // transverse path length sPerp; tsPar keeps the parameters at the previous
      // hit, which the jacobian needs. The crossing is the Hermite cubic solved
      // onto the plane, not a helix solution: within 10 nm of the exact crossing
      // for 97 % of hits (measured 2026-09).
      propPar.aij(0, 0) = tsXyz.x;
      propPar.aij(1, 0) = tsXyz.y;
      propPar.aij(2, 0) = tsXyz.z;
      propPar.aij(3, 0) = tsXyz.inv_pt;
      propPar.aij(4, 0) = Matriplex::atan2_poly(tsXyz.py, tsXyz.px);
      propPar.aij(5, 0) = tsXyz.theta;

      sPerp = tsXyz.dalpha / (tsXyz.inv_pt * tsXyz.inv_k);
    }

    void do_kalman_stuff();
  };

}  // namespace mkfit

#endif
