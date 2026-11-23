#ifndef RecoTracker_MkFitCore_src_MkBins_h
#define RecoTracker_MkFitCore_src_MkBins_h

#include "RecoTracker/MkFitCore/src/MiniPropagators.h"
#include "RecoTracker/MkFitCore/src/V2p2Config.h"

namespace mkfit {

  class LayerInfo;
  class LayerOfHits;
  struct MkRZLimits;

  // Position block of the track covariance at MkBins::m_sp2, filled by
  // MkBins::transport_position_cov().
  struct MkBinTrackCovExtract {
    MPlexQF m_cov_0_0 = {0.0f};
    MPlexQF m_cov_0_1 = {0.0f};
    MPlexQF m_cov_1_1 = {0.0f};
    MPlexQF m_cov_2_2 = {0.0f};
    // Read by the surface references of dq, which act on the whole position block.
    MPlexQF m_cov_0_2 = {0.0f};
    MPlexQF m_cov_1_2 = {0.0f};

    MPlexQF calc_err_xy(const MPlexQF& x, const MPlexQF& y) const {
      return x * x * m_cov_0_0 + y * y * m_cov_1_1 + 2.0f * x * y * m_cov_0_1;
    };
  };

  //============================================================================

  struct MkBinLimits {
    MPlexQUH q1, q2, p1, p2;
  };

  //============================================================================

  struct MkBins {
    mini_propagators::InitialStatePlex m_isp;
    // Crossings of the layer's two bounding surfaces in the search direction:
    // m_sp1 the near one, m_sp2 the far one. See prop_to_limits_in_order().
    mini_propagators::StatePlex m_sp1, m_sp2;

    MPlexQF m_phi1, m_phi2;  // phi at m_sp1 and m_sp2, from determine_bin_windows()
    MPlexQF m_phi_min, m_phi_max, m_phi_center, m_phi_delta;
    MPlexQF m_q_min, m_q_max;

    MPlexQF m_dphi_track, m_dq_track;  // 3 sigma track errors, from the covariance at m_sp2

    int m_n_proc;
    bool m_is_barrel;  // set in prop_to_limits_in_order()

    // -----------------------------------------------------

    MkBins(int n_proc) : m_n_proc(n_proc) {}

    void prop_to_limits_in_order(const MkRZLimits& ls);

    // Covariance position block at m_sp2 from the previous-hit state (par0, err0).
    void transport_position_cov(const MPlexLV& par0, const MPlexLS& err0, MkBinTrackCovExtract& tce) const;

    void determine_bin_windows(const MkBinTrackCovExtract& cov_ex);
#if defined(MKFIT_STANDALONE)
    void surface_reference_dq(const MkBinTrackCovExtract& cov_ex);
#endif

    // Fetch derived from the v2p2 cut, Config::V2p2::Window.
    void find_bin_ranges(const LayerOfHits& loh, MkBinLimits& bl);
  };

}  // namespace mkfit

#endif
