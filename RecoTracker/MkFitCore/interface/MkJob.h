#ifndef RecoTracker_MkFitCore_interface_MkJob_h
#define RecoTracker_MkFitCore_interface_MkJob_h

#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

namespace mkfit {

  class MkJob {
  public:
    const TrackerInfo &m_trk_info;
    // Config &config; // If we want to get rid of namespace / global config
    const IterationConfig &m_iter_config;
    const EventOfHits &m_event_of_hits;
    const BeamSpot &m_beam_spot;

    const IterationMaskIfcBase *m_iter_mask_ifc = nullptr;

    bool m_in_fwd = true;
    void switch_to_backward() { m_in_fwd = false; }

    int num_regions() const { return m_iter_config.m_n_regions; }
    const auto regions_begin() const { return m_iter_config.m_region_order.begin(); }
    const auto regions_end() const { return m_iter_config.m_region_order.end(); }

    const auto &steering_params(int i) { return m_iter_config.m_steering_params[i]; }

    const auto &params() const { return m_iter_config.m_params; }
    const auto &params_bks() const { return m_iter_config.m_backward_params; }
    const auto &params_cur() const { return m_in_fwd ? params() : params_bks(); }

    int max_max_cands() const { return std::max(params().maxCandsPerSeed, params_bks().maxCandsPerSeed); }

    const std::vector<bool> *get_mask_for_layer(int layer) {
      return m_iter_mask_ifc ? m_iter_mask_ifc->get_mask_for_layer(layer) : nullptr;
    }
  };

}  // namespace mkfit

#endif
