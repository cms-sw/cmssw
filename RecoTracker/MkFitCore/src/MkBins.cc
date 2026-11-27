#include "RecoTracker/MkFitCore/src/MkBins.h"
#include "RecoTracker/MkFitCore/src/MkRZLimits.h"

#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include <algorithm>
#include <cmath>

namespace mkfit {

  using namespace Config::V2p2;

  namespace mp = mini_propagators;

  //==============================================================================

  void MkBins::prop_to_limits_in_order(const MkRZLimits& ls) {
    // m_isp is at the previous hit. m_sp1 is the crossing of the near bounding
    // surface of the layer, m_sp2 of the far one; m_isp is re-initialised at
    // m_sp1 on the way. dalpha accumulates, so both are measured from the
    // previous hit, which transport_position_cov() relies on.
    m_is_barrel = ls.m_is_barrel;
    if (m_is_barrel) {
      float r;
      r = ls.m_is_outward ? ls.m_rin : ls.m_rout;
      m_isp.propagate_to_r(mp::PA_Exact, r, m_sp1, true, m_n_proc);
      m_isp = m_sp1;
      r = ls.m_is_outward ? ls.m_rout : ls.m_rin;
      m_isp.propagate_to_r(mp::PA_Exact, r, m_sp2, true, m_n_proc);
    } else {
      float z1 = ls.m_zmin, z2 = ls.m_zmax;
      bool is_pos = ls.m_layer_info_1->layer_type() == LayerInfo::EndCapPos;
      if ((is_pos && !ls.m_is_outward) || (!is_pos && ls.m_is_outward))
        std::swap(z1, z2);

      m_isp.propagate_to_z(mp::PA_Exact, z1, m_sp1, true, m_n_proc);
      m_isp = m_sp1;
      m_isp.propagate_to_z(mp::PA_Exact, z2, m_sp2, true, m_n_proc);
    }
  }

  //----------------------------------------------------------------------------
  // transport_position_cov() -- position block of the covariance at m_sp2,
  // J C0 J^T with J = [P_in | dx/d(ipt, phi, theta)] at fixed path length,
  // curvilinear at both ends. See doc/MkFinderV2p2-DesignNotes.md, "Track
  // covariance at the layer".
  //----------------------------------------------------------------------------

  void MkBins::transport_position_cov(const MPlexLV& par0, const MPlexLS& err0, MkBinTrackCovExtract& tce) const {
    // All NN lanes; the caller fills unused ones with a valid state.
    using MPF = MPlexQF;

    const MPF ipt = par0.ReduceFixedIJ(3, 0);
    const MPF iptinv = 1.0f / ipt;
    MPF sp, cp, st, ct;
    Matriplex::fast_sincos(par0.ReduceFixedIJ(4, 0), sp, cp);
    Matriplex::fast_sincos(par0.ReduceFixedIJ(5, 0), st, ct);
    const MPF cot = ct / st;

    const MPF& a = m_sp2.dalpha;
    const MPF k = 1.0f / m_isp.inv_k;
    const MPF dx = m_sp2.x - par0.ReduceFixedIJ(0, 0);
    const MPF dy = m_sp2.y - par0.ReduceFixedIJ(1, 0);
    const MPF ak = a * k;

    // f1 = a cos a - sin a, f2 = a sin a - (1 - cos a), by series below |a| = 0.25.
    const MPF a2 = a * a;
    MPF f1 = a * a2 * (-1.0f / 3.0f + a2 * (1.0f / 30.0f - a2 * (1.0f / 840.0f)));
    MPF f2 = a2 * (0.5f - a2 * (1.0f / 8.0f - a2 * (1.0f / 144.0f)));
    for (int i = 0; i < m_n_proc; ++i) {
      if (std::abs(a[i]) >= 0.25f) {
        const float sa = std::sin(a[i]), ca = std::cos(a[i]);
        f1[i] = a[i] * ca - sa;
        f2[i] = a[i] * sa - (1.0f - ca);
      }
    }
    const MPF kpx = k * cp * iptinv * iptinv;
    const MPF kpy = k * sp * iptinv * iptinv;

    // J = [P_in | D], 3 x 6, over (x, y, z, ipt, phi, theta). P_in = 1 - n0 n0^T.
    const MPF n0[3] = {cp * st, sp * st, ct};
    MPF J[3][6];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        J[r][c] = (r == c ? 1.0f : 0.0f) - n0[r] * n0[c];
    J[0][3] = kpx * f1 - kpy * f2;
    J[0][4] = -dy;
    J[0][5] = ak * m_sp2.px * cot;
    J[1][3] = kpy * f1 + kpx * f2;
    J[1][4] = dx;
    J[1][5] = ak * m_sp2.py * cot;
    J[2][3] = 0.0f;
    J[2][4] = 0.0f;
    J[2][5] = -ak * iptinv;

    // M = J C0, then C = M J^T.
    MPF E[6][6];
    for (int m = 0; m < 6; ++m)
      for (int c = m; c < 6; ++c)
        E[m][c] = E[c][m] = err0.ReduceFixedIJ(m, c);
    MPF M[3][6];
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 6; ++c) {
        M[r][c] = J[r][0] * E[0][c];
        for (int m = 1; m < 6; ++m)
          M[r][c] += J[r][m] * E[m][c];
      }
    MPF C[3][3];
    for (int r = 0; r < 3; ++r)
      for (int c = r; c < 3; ++c) {
        C[r][c] = M[r][0] * J[c][0];
        for (int m = 1; m < 6; ++m)
          C[r][c] += M[r][m] * J[c][m];
        C[c][r] = C[r][c];
      }

    // Project onto the plane normal to the momentum at m_sp2: P C P.
    const MPF pinv = 1.0f / Matriplex::sqrt(m_sp2.px * m_sp2.px + m_sp2.py * m_sp2.py + m_sp2.pz * m_sp2.pz);
    const MPF n1[3] = {m_sp2.px * pinv, m_sp2.py * pinv, m_sp2.pz * pinv};
    MPF Cn[3];
    for (int r = 0; r < 3; ++r)
      Cn[r] = C[r][0] * n1[0] + C[r][1] * n1[1] + C[r][2] * n1[2];
    const MPF nCn = n1[0] * Cn[0] + n1[1] * Cn[1] + n1[2] * Cn[2];
    auto pcp = [&](int r, int c) { return C[r][c] - n1[r] * Cn[c] - Cn[r] * n1[c] + n1[r] * n1[c] * nCn; };

    tce.m_cov_0_0 = pcp(0, 0);
    tce.m_cov_0_1 = pcp(0, 1);
    tce.m_cov_1_1 = pcp(1, 1);
    tce.m_cov_2_2 = pcp(2, 2);
    tce.m_cov_0_2 = pcp(0, 2);
    tce.m_cov_1_2 = pcp(1, 2);
  }

  //----------------------------------------------------------------------------

  void MkBins::determine_bin_windows(const MkBinTrackCovExtract& cov_ex) {
    m_phi1 = Matriplex::atan2_poly(m_sp1.y, m_sp1.x);
    m_phi2 = Matriplex::atan2_poly(m_sp2.y, m_sp2.x);
    Matriplex::min_max(m_phi1, m_phi2, m_phi_min, m_phi_max);
    m_phi_delta = m_phi_max - m_phi_min;
    m_phi_center = 0.5f * (m_phi_max + m_phi_min);
    for (int ii = 0; ii < NN; ++ii) {
      if (ii < m_n_proc) {
        if (m_phi_delta[ii] > Const::PI) {
          std::swap(m_phi_max[ii], m_phi_min[ii]);
          m_phi_delta[ii] = Const::TwoPI - m_phi_delta[ii];
          m_phi_center[ii] = squashPhiMinimal(m_phi_center[ii] + Const::PI);
        }
        m_phi_delta[ii] *= 0.5f;
      }
    }

    // Calculate dphi_track, dq_track differs for barrel/endcap
    MPlexQF r2_c = m_isp.x * m_isp.x + m_isp.y * m_isp.y;
    MPlexQF r2inv_c = 1.0f / r2_c;

    // sigma_phi = J sigma_xy J^T with J = (-y, x)/r^2, so |J| = 1/r. J is taken at
    // the layer crossing with the smaller radius, where sigma_phi is largest. See
    // doc/MkFinderV2p2-DesignNotes.md, "Search window".
    MPlexQF jx, jy, r2inv_j;
    {
      const MPlexQF r2_1 = m_sp1.x * m_sp1.x + m_sp1.y * m_sp1.y;
      const MPlexQF r2_2 = m_sp2.x * m_sp2.x + m_sp2.y * m_sp2.y;
      for (int i = 0; i < m_n_proc; ++i) {
        const bool first = r2_1[i] <= r2_2[i];
        jx[i] = first ? m_sp1.x[i] : m_sp2.x[i];
        jy[i] = first ? m_sp1.y[i] : m_sp2.y[i];
        r2inv_j[i] = 1.0f / (first ? r2_1[i] : r2_2[i]);
      }
    }
    MPlexQF dphidx_c = -jy * r2inv_j;
    MPlexQF dphidy_c = jx * r2inv_j;
    m_dphi_track = 3.0f * cov_ex.calc_err_xy(dphidx_c, dphidy_c).abs().sqrt();

    if (m_is_barrel) {
      Matriplex::min_max(m_sp1.z, m_sp2.z, m_q_min, m_q_max);
      m_dq_track = 3.0f * Matriplex::abs(cov_ex.m_cov_2_2).sqrt();
    } else {
      Matriplex::min_max(Matriplex::hypot(m_sp1.x, m_sp1.y), Matriplex::hypot(m_sp2.x, m_sp2.y), m_q_min, m_q_max);
      m_dq_track = 3.0f * (r2inv_c * cov_ex.calc_err_xy(m_isp.x, m_isp.y).abs()).sqrt();
    }

#if defined(MKFIT_STANDALONE)
    if (Diag::mkbins_surface_q)
      surface_reference_dq(cov_ex);
#endif
  }

  void MkBins::find_bin_ranges(const LayerOfHits& loh, MkBinLimits& bl, float dphi_trk_fac, float dq_trk_fac) {
    if (!(dphi_trk_fac > 0.0f))
      dphi_trk_fac = Window::dphi_trk_fac;
    if (!(dq_trk_fac > 0.0f))
      dq_trk_fac = Window::dq_trk_fac;
    for (int i = 0; i < NN; ++i) {
      if (i < m_n_proc) {
        // Fetch what the cut can accept, using the layer's largest hit extent since
        // the hit is not known yet, then extend by whole bins on the bin index.
        // The margin is capped at Window::dphi_max_rad.
        // See doc/MkFinderV2p2-DesignNotes.md, "Search window".
        const float phi_hit_term =
            Window::phi_per_hit ? Window::dphi_hit_fac * loh.max_hit_phi_half_extent() : Window::dphi_flat_rad;
        const float cut_dphi = Window::capped_dphi(dphi_trk_fac * m_dphi_track[i] + phi_hit_term);
        auto pr = loh.phiRangeBins(m_phi_min[i] - cut_dphi, m_phi_max[i] + cut_dphi);
        bl.p1[i] = loh.phiMaskApply(pr.begin - Window::phi_extra_bins);
        bl.p2[i] = loh.phiMaskApply(pr.end + Window::phi_extra_bins);

        // Same for q. The q axis is bounded, so the extension clamps.
        const float cut_dq = dq_trk_fac * m_dq_track[i] + Window::dq_hit_fac * loh.max_hit_q_half_length();
        auto qr = loh.qRangeBins(m_q_min[i] - cut_dq, m_q_max[i] + cut_dq);
        const int nq = (int)loh.qNBins();
        int qb = (int)qr.begin - Window::q_extra_bins;
        int qe = (int)qr.end + Window::q_extra_bins;
        bl.q1[i] = (unsigned short)(qb < 0 ? 0 : qb);
        bl.q2[i] = (unsigned short)(qe > nq ? nq : qe);
      }
    }
  }

}  // namespace mkfit
