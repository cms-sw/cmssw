#ifndef RecoTracker_MkFitCore_src_MkFinderV2p2_h
#define RecoTracker_MkFitCore_src_MkFinderV2p2_h

#include "RecoTracker/MkFitCore/interface/SteeringParams.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"
#include "RecoTracker/MkFitCore/interface/MkJob.h"

#include "MkRZLimits.h"
#include "MkFinderV2p2Structures.h"

#include "MkBins.h"

#include <list>
#include <vector>

// Per-layer policy counters, standalone only: see standalone/V2p2Diag.h. The
// finder increments them through V2P2_COUNT() and V2P2_COUNT_ADD(), which expand
// to nothing in the CMSSW build.
#if defined(MKFIT_STANDALONE)
#include "RecoTracker/MkFitCore/standalone/V2p2Diag.h"
#else
#define V2P2_COUNT(field) ((void)0)
#define V2P2_COUNT_ADD(field, n) ((void)0)
#endif

namespace mkfit {

  // Switches are in V2p2Config.h, namespace Config::V2p2; the per-iteration
  // tunings are V2p2Params (interface/V2p2Params.h), read through MkJob::v2p2().

  class Event;

  // The v2p2 track finder: runs a block of seeds through the layer plan, one
  // layer at a time. MkBuilder::findTracksStandardv2p2() drives it: setup(),
  // awaken_candidates(), then begin_layer() / process_layer() / end_layer() per
  // plan entry; the pool's recycler calls reset(). How the pieces fit together: the overview at the
  // top of MkFinderV2p2.cc. See doc/MkFinderV2p2-DesignNotes.md.
  class MkFinderV2p2 {
  public:
    MkFinderV2p2() = default;

    void setup(const MkJob* job,
               EventOfCombCandidates& eoccs,
               int seed_begin,
               int seed_end,
               SteeringParams::iterator& sp_it,
               const Event* ev);
    void reset();

    // Track terms of the search window: the iteration's V2p2Params, else the built-in ones.
    float win_dphi_trk_fac() const;
    float win_dq_trk_fac() const;

    int awaken_candidates();

    void begin_layer();
    void process_layer();
    void end_layer();

  private:
    //--------------------------------------------------------------------------
    // The block of seeds: their CombCandidates in the EventOfCombCandidates, and
    // how many are dormant (not yet picked up) and finding.
    class BatchManager {
      friend class MkFinderV2p2;

      EventOfCombCandidates* mp_eoccs = nullptr;
      int m_begin;
      int m_end;

      int m_n_dormant;
      int m_n_finding;

    public:
      CombCandidate* ccand_ptr(int i) const { return &(*mp_eoccs)[i]; }

      void setup(EventOfCombCandidates& eoccs, int seed_begin, int seed_end) {
        mp_eoccs = &eoccs;
        m_begin = seed_begin;
        m_end = seed_end;
        m_n_dormant = 0;
        m_n_finding = 0;
      }
      void release() { mp_eoccs = nullptr; }

      int n_total() const { return m_end - m_begin; }
      int n_dormant() const { return m_n_dormant; }
      int n_finding() const { return m_n_finding; }
      bool has_dormant_ccands() const { return n_dormant() > 0; }

      // Iteration over all CombCandidates of the block.
      class iterator {
        CombCandidate* m_ccand;

      public:
        iterator(CombCandidate* bm) : m_ccand(bm) {}
        CombCandidate& operator*() { return *m_ccand; }
        iterator& operator++() {
          ++m_ccand;
          return *this;
        }
        bool operator!=(const iterator& i) const { return m_ccand != i.m_ccand; }
      };

      iterator begin() const { return iterator(ccand_ptr(m_begin)); }
      iterator end() const { return iterator(ccand_ptr(m_end - 1) + 1); }
    };

  public:
    // For debug printouts in MkBuilder.
    const BatchManager& batch_mgr() const { return m_batch_mgr; }

  private:
    //--------------------------------------------------------------------------
    // Per-pass state of process_layer_batch(). LayerBatch is NN candidates wide
    // and indexed by i; HitBatch is NN (candidate, hit) pairs wide and indexed by
    // h, with cand_idcs[h] naming the candidate.
    struct LayerBatch {
      int N_proc = 0;
      CandIncoming* cand[NN];         // the candidates in this pass
      MkBins B{0};                    // isp + the two bounding-surface crossings
      MkBinTrackCovExtract TCE;       // position block of the window covariance
      MkBinLimits BL_p, BL_s;         // binnor ranges, primary / secondary layer
      mini_propagators::Hermite3D H;  // cubic through sp1, sp2 -- the trajectory model
      // Local hit density for the likelihood score: the hits counted in the cut
      // region, and the region's padding, max over the sub-layers.
      int n_scanned[NN] = {0};
      float rho_dphi[NN] = {0.0f};
      float rho_dq[NN] = {0.0f};
#ifdef MKFIT_TRACE
      int tr_layersearch_ids[NN];
#endif
    };

    struct HitBatch {
      int fill_pos = 0;
      MPlexQI cand_idcs;       // -> LayerBatch::cand
      MPlexQUI hit_idcs;       // index within the LayerOfHits
      MPlexQUI hit_orig_idcs;  // index in the layer's hit vector, as HitOnTrack::index
      mini_propagators::InitialStatePlex is_plex;
#ifdef MKFIT_TRACE_PROP_COMPARE
      mini_propagators::StatePlex h_plex;  // PA_Line cross-check against the Hermite
#endif
    };

    // One competitor in the end-of-layer selection.
    struct SelEntry {
      int tcand_idx;  // the TrackCand of the CombCandidate it continues
      int node_idx;   // arena index of its in-layer path; -1 = it took no hit here
      int fake_hit;   // with add_fake, the fake HoT to record: miss, edge, gap or stop
      bool add_fake;  // false: nothing recorded (layer out of reach, or already stopped)
      float score;    // its score after this layer
    };

    //--------------------------------------------------------------------------
    // Layer control: pull CombCandidates into the layer and run the queue
    // through process_layer_batch(), NN candidates at a time.
    bool any_Ccreps_to_begin() const { return m_active_ccreps_pos != m_active_ccreps.end(); }
    void begin_next_Ccrep_in_layer();

    int cand_queue_size() const { return (int)m_cand_queue.size() - m_cand_queue_head; }
    bool enough_work_for_batch() const { return cand_queue_size() >= NN; }
    bool any_work_for_batch() const { return cand_queue_size() > 0; }
    void process_layer_batch();

    // Candidate-stopping cuts applied at pull-in, where only the candidate's own
    // state is needed. Returns the reason, or SR_NotStopped to keep going.
    TrackCand::StopReason_e stop_cuts_at_pickup(const TrackCand& tc) const;
    bool long_step_to_layer(const TrackCand& tc) const;
    // Which fake HoT a candidate that took no hit in this layer gets: the hole
    // limits, the WSR override, and the gap case, in V1's order.
    int fake_hit_index(const TrackCand& tc, const WSR_Result& wsr) const;

    //--------------------------------------------------------------------------
    // The phases of process_layer_batch(), in order; determine_wsr() is called
    // from determine_search_windows().
    void prop_to_layer_edges(LayerBatch& b);
    void determine_search_windows(LayerBatch& b);
    void determine_wsr(LayerBatch& b);
    void select_hits(LayerBatch& b);
    void preselect_hit_batch(LayerBatch& b, HitBatch& hb, const LayerOfHits& L, int N_proc_hits, bool is_sec_layer);
    // The track's q error referenced to the hit's module plane. See MkFinderV2p2.cc.
    static float surface_referenced_dq(float dq_track_fallback,
                                       const MkBinTrackCovExtract& TCE,
                                       int pi,
                                       const mini_propagators::StatePlex& h3_state,
                                       int h,
                                       const MPlex3V& module_norm,
                                       bool is_barrel);
    void order_hits_along_path(LayerBatch& b);

    //--------------------------------------------------------------------------
    // The in-layer combinatorial search: expand_in_layer() grows the CandNode
    // tree. Depth 0 (the Hermite crossing is known, sPerp path) and deeper steps
    // (the plane is solved) need different propagation, so there are two Kalman
    // batches. Each fires only when full; rk_drain() empties both at end of layer.
    // rk_ = running Kalman: these batches and the functions that run them.
    void expand_in_layer(LayerBatch& b);
    void rk_setup(KalmanOpArgs& koa, bool solve_plane);
    void rk_add_item(
        KalmanOpArgs& koa, CandIncoming& cand, const CandIncoming::PassedHitMatch& phm, int parent_idx, int hit_pos);
    void rk_flush(KalmanOpArgs& koa);
    void rk_expand_node(int ni);
    void rk_pump();
    void rk_drain();
    void load_hit_module(KalmanOpArgs& koa, const CandIncoming::PassedHitMatch& phm) const;
    // Turn the Kalman results accumulated in m_kalman_out into arena nodes, keeping
    // those that pass the chi2 cut. Their indices go to m_nodes_harvested.
    void harvest_nodes();
    // A slot for one node: from the free list if it has any, else appended.
    int node_slot();
    // The direction- and layer-dependent part of a layer step, filled once per
    // path root and carried down the tree, and once per hole entry.
    void fill_step_geometry(LayerStepFeatures& f, float log_rho) const;

    //--------------------------------------------------------------------------
    // End-of-layer selection: select_and_materialise() ranks the paths, holes and
    // untouched candidates of one CombCandidate and registers the survivors.
    // After a layer batch: select each CombCandidate of the batch, or of
    // m_ccrep_ready, that has nothing queued or in flight, and free its nodes.
    void select_completed_ccreps(const LayerBatch& b);
    void select_and_materialise(CCandRep& ccrep);
    void drop_sister_dominated_holes(const CombCandidate& ccand);
    void offer_best_short(CombCandidate& ccand, const TrackCand& tc) const;

    //==========================================================================
    // Data

    // The job, the layer plan position and the event, for one seed block.
    const MkJob* mp_job = nullptr;
    SteeringParams::iterator* mp_steeringparams_iter = nullptr;
    const Event* mp_event = nullptr;

    BatchManager m_batch_mgr;

    // Per-layer state and control.
    std::list<CCandRep> m_active_ccreps;
    // Next CombCandidate to be pulled into the layer by begin_next_Ccrep_in_layer().
    std::list<CCandRep>::iterator m_active_ccreps_pos;
    // Pre-selection queue: the incoming candidates waiting for propagation, binnor fetch
    // and hit pre-selection. A FIFO as a vector and a read position, reset when it
    // drains, so it keeps its capacity.
    std::vector<CandIncoming*> m_cand_queue;
    int m_cand_queue_head = 0;
    MkRZLimits m_rz_limits;

    // The in-layer combinatorial tree: one arena per finder, reached by index,
    // cleared with capacity kept at end of layer.
    std::vector<CandNode> m_node_arena;
    // Slots of nodes whose CombCandidate has been selected, reused before the
    // arena grows. A node is freed only with its whole CombCandidate, so no live
    // node can have a freed parent.
    std::vector<int> m_node_free;
    // Arena indices appended by the last harvest_nodes().
    std::vector<int> m_nodes_harvested;
    // The two Kalman batches of expand_in_layer(), and the nodes whose children
    // are not yet queued.
    KalmanOpArgs m_koa_d0, m_koa_deep;
    std::vector<int> m_deep_queue;
    // Kalman outcomes of the last flush, drained into the arena by
    // harvest_nodes().
    std::vector<KalmanOpArgs::ItemOut> m_kalman_out;
    // CombCandidates whose last in-flight work has finished, for
    // select_completed_ccreps(); m_ccrep_touched is rk_flush()'s scratch list.
    std::vector<CCandRep*> m_ccrep_ready;
    std::vector<CCandRep*> m_ccrep_touched;

    // End-of-layer selection: the competitors, and the survivors, built before
    // the CombCandidate is rewritten from them.
    std::vector<SelEntry> m_sel;
    std::vector<char> m_has_cand;  // per TrackCand: entered as a CandIncoming
    std::vector<TrackCand> m_new_cands;
  };

}  // end namespace mkfit

#endif
