#ifndef RecoTracker_MkFitCore_src_CandCloner_h
#define RecoTracker_MkFitCore_src_CandCloner_h

#include "MkFinder.h"

#include <vector>

namespace mkfit {

  class IterationParams;
  class EventOfCombCandidates;

  //#define CC_TIME_LAYER
  //#define CC_TIME_ETA

  class CandCloner {
  public:
    // Maximum number of seeds processed in one call to processSeedRange()
    static const int s_max_seed_range = MPT_SIZE;

    CandCloner() { t_cands_for_next_lay.resize(s_max_seed_range); }

    void setup(const IterationParams &ip);
    void release();

    void begin_eta_bin(EventOfCombCandidates *e_o_ccs,
                       std::vector<UpdateIndices> *update_list,
                       std::vector<std::vector<TrackCand>> *extra_cands,
                       int start_seed,
                       int n_seeds);
    void begin_layer(int lay);
    void begin_iteration();

    void add_cand(int idx, const IdxChi2List &cand_info) {
      m_hits_to_add[idx].push_back(cand_info);

      m_idx_max = std::max(m_idx_max, idx);
    }

    int num_cands(int idx) { return m_hits_to_add[idx].size(); }

    void end_iteration();
    void end_layer();
    void end_eta_bin();

    void doWork(int idx);

    void processSeedRange(int is_beg, int is_end);

    // Accessor for MkFitter
    CombCandidate &combCandWithOriginalIndex(int idx) { return mp_event_of_comb_candidates->cand(idx); }

  private:
    int m_idx_max, m_idx_max_prev;
    std::vector<std::vector<IdxChi2List>> m_hits_to_add;

    const IterationParams *mp_iteration_params = nullptr;
    EventOfCombCandidates *mp_event_of_comb_candidates;
    std::vector<UpdateIndices> *mp_kalman_update_list;
    std::vector<std::vector<TrackCand>> *mp_extra_cands;

#if defined(CC_TIME_ETA) or defined(CC_TIME_LAYER)
    double t_eta, t_lay;
#endif

    int m_start_seed, m_n_seeds;
    int m_layer;

    // Temporary in processSeedRange(), resized/reserved  in constructor.
    // Size of this one is s_max_seed_range
    std::vector<std::vector<TrackCand>> t_cands_for_next_lay;
  };

}  // end namespace mkfit
#endif
