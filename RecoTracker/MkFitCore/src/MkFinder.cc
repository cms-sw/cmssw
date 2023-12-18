#include "MkFinder.h"

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "CandCloner.h"
#include "FindingFoos.h"
#include "KalmanUtilsMPlex.h"
#include "MatriplexPackers.h"
#include "MiniPropagators.h"

//#define DEBUG
#include "Debug.h"

#if defined(MKFIT_STANDALONE)
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

#ifdef RNT_DUMP_MkF_SelHitIdcs
// declares struct RntIfc_selectHitIndices rnt_shi in unnamed namespace;
#include "RecoTracker/MkFitCore/standalone/RntDumper/MkFinder_selectHitIndices.icc"
#endif

#include "vdt/atan2.h"

#include <algorithm>
#include <queue>

namespace mkfit {

  void MkFinder::setup(const PropagationConfig &pc,
                       const IterationConfig &ic,
                       const IterationParams &ip,
                       const IterationLayerConfig &ilc,
                       const SteeringParams &sp,
                       const std::vector<bool> *ihm,
                       const Event *ev,
                       int region,
                       bool infwd) {
    m_prop_config = &pc;
    m_iteration_config = &ic;
    m_iteration_params = &ip;
    m_iteration_layer_config = &ilc;
    m_steering_params = &sp;
    m_iteration_hit_mask = ihm;
    m_event = ev;
    m_current_region = region;
    m_in_fwd = infwd;
  }

  void MkFinder::setup_bkfit(const PropagationConfig &pc, const SteeringParams &sp, const Event *ev) {
    m_prop_config = &pc;
    m_steering_params = &sp;
    m_event = ev;
  }

  void MkFinder::release() {
    m_prop_config = nullptr;
    m_iteration_config = nullptr;
    m_iteration_params = nullptr;
    m_iteration_layer_config = nullptr;
    m_steering_params = nullptr;
    m_iteration_hit_mask = nullptr;
    m_event = nullptr;
    m_current_region = -1;
    m_in_fwd = true;
  }

  void MkFinder::begin_layer(const LayerOfHits &layer_of_hits) {
#ifdef RNT_DUMP_MkF_SelHitIdcs
    const LayerOfHits &L = layer_of_hits;
    const LayerInfo &LI = *L.layer_info();
    rnt_shi.ResetH();
    rnt_shi.ResetF();
    *rnt_shi.h = {m_event->evtID(),
                  m_iteration_config->m_iteration_index,
                  m_iteration_config->m_track_algorithm,
                  m_current_region,
                  L.layer_id(),
                  L.is_barrel() ? LI.rin() : LI.zmin(),
                  LI.is_barrel() ? LI.rout() : LI.zmax(),
                  L.is_barrel(),
                  L.is_pixel(),
                  L.is_stereo()};
    *rnt_shi.f = *rnt_shi.h;
#endif
  }

  void MkFinder::end_layer() {
#ifdef RNT_DUMP_MkF_SelHitIdcs
    rnt_shi.FillH();
    rnt_shi.FillF();
#endif
  }

  //==============================================================================
  // Input / Output TracksAndHitIdx
  //==============================================================================

  void MkFinder::inputTracksAndHitIdx(const std::vector<Track> &tracks, int beg, int end, bool inputProp) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    const int iI = inputProp ? iP : iC;

    for (int i = beg, imp = 0; i < end; ++i, ++imp) {
      copy_in(tracks[i], imp, iI);
    }
  }

  void MkFinder::inputTracksAndHitIdx(
      const std::vector<Track> &tracks, const std::vector<int> &idxs, int beg, int end, bool inputProp, int mp_offset) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    const int iI = inputProp ? iP : iC;

    for (int i = beg, imp = mp_offset; i < end; ++i, ++imp) {
      copy_in(tracks[idxs[i]], imp, iI);
    }
  }

  void MkFinder::inputTracksAndHitIdx(const std::vector<CombCandidate> &tracks,
                                      const std::vector<std::pair<int, int>> &idxs,
                                      int beg,
                                      int end,
                                      bool inputProp) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    const int iI = inputProp ? iP : iC;

    for (int i = beg, imp = 0; i < end; ++i, ++imp) {
      const TrackCand &trk = tracks[idxs[i].first][idxs[i].second];

      copy_in(trk, imp, iI);

      m_SeedIdx(imp, 0, 0) = idxs[i].first;
      m_CandIdx(imp, 0, 0) = idxs[i].second;
      m_SeedOriginIdx[imp] = tracks[idxs[i].first].seed_origin_index();
    }
  }

  void MkFinder::inputTracksAndHits(const std::vector<CombCandidate> &tracks,
                                    const LayerOfHits &layer_of_hits,
                                    const std::vector<UpdateIndices> &idxs,
                                    int beg,
                                    int end,
                                    bool inputProp) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    const int iI = inputProp ? iP : iC;

    for (int i = beg, imp = 0; i < end; ++i, ++imp) {
      const TrackCand &trk = tracks[idxs[i].seed_idx][idxs[i].cand_idx];

      copy_in(trk, imp, iI);

      m_SeedIdx(imp, 0, 0) = idxs[i].seed_idx;
      m_CandIdx(imp, 0, 0) = idxs[i].cand_idx;
      m_SeedOriginIdx[imp] = tracks[idxs[i].seed_idx].seed_origin_index();

      // Reuse selectHitIndices() arrays -- used also in packModuleNormDir()
      m_XHitArr(imp, 0, 0) = idxs[i].hit_idx;
      m_XHitSize(imp, 0, 0) = 1;

      const Hit &hit = layer_of_hits.refHit(idxs[i].hit_idx);
      m_msErr.copyIn(imp, hit.errArray());
      m_msPar.copyIn(imp, hit.posArray());
    }
  }

  void MkFinder::inputOverlapHits(const LayerOfHits &layer_of_hits,
                                  const std::vector<UpdateIndices> &idxs,
                                  int beg,
                                  int end) {
    // Copy overlap hit values in.

    for (int i = beg, imp = 0; i < end; ++i, ++imp) {
      const Hit &hit = layer_of_hits.refHit(idxs[i].ovlp_idx);
      m_msErr.copyIn(imp, hit.errArray());
      m_msPar.copyIn(imp, hit.posArray());
    }
  }

  void MkFinder::inputTracksAndHitIdx(const std::vector<CombCandidate> &tracks,
                                      const std::vector<std::pair<int, IdxChi2List>> &idxs,
                                      int beg,
                                      int end,
                                      bool inputProp) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    const int iI = inputProp ? iP : iC;

    for (int i = beg, imp = 0; i < end; ++i, ++imp) {
      const TrackCand &trk = tracks[idxs[i].first][idxs[i].second.trkIdx];

      copy_in(trk, imp, iI);

      m_SeedIdx(imp, 0, 0) = idxs[i].first;
      m_CandIdx(imp, 0, 0) = idxs[i].second.trkIdx;
      m_SeedOriginIdx[imp] = tracks[idxs[i].first].seed_origin_index();
    }
  }

  void MkFinder::outputTracksAndHitIdx(std::vector<Track> &tracks, int beg, int end, bool outputProp) const {
    // Copies requested track parameters into Track objects.
    // The tracks vector should be resized to allow direct copying.

    const int iO = outputProp ? iP : iC;

    for (int i = beg, imp = 0; i < end; ++i, ++imp) {
      copy_out(tracks[i], imp, iO);
    }
  }

  void MkFinder::outputTracksAndHitIdx(
      std::vector<Track> &tracks, const std::vector<int> &idxs, int beg, int end, bool outputProp) const {
    // Copies requested track parameters into Track objects.
    // The tracks vector should be resized to allow direct copying.

    const int iO = outputProp ? iP : iC;

    for (int i = beg, imp = 0; i < end; ++i, ++imp) {
      copy_out(tracks[idxs[i]], imp, iO);
    }
  }

  void MkFinder::packModuleNormDir(
      const LayerOfHits &layer_of_hits, int hit_cnt, MPlexHV &norm, MPlexHV &dir, int N_proc) const {
    for (int itrack = 0; itrack < N_proc; ++itrack) {
      if (hit_cnt < m_XHitSize[itrack]) {
        const auto &hit = layer_of_hits.refHit(m_XHitArr.constAt(itrack, hit_cnt, 0));
        unsigned int mid = hit.detIDinLayer();
        const ModuleInfo &mi = layer_of_hits.layer_info()->module_info(mid);
        norm.At(itrack, 0, 0) = mi.zdir[0];
        norm.At(itrack, 1, 0) = mi.zdir[1];
        norm.At(itrack, 2, 0) = mi.zdir[2];
        dir.At(itrack, 0, 0) = mi.xdir[0];
        dir.At(itrack, 1, 0) = mi.xdir[1];
        dir.At(itrack, 2, 0) = mi.xdir[2];
      }
    }
  }

  //==============================================================================
  // getHitSelDynamicWindows
  //==============================================================================
  // From HitSelectionWindows.h: track-related config on hit selection windows

  void MkFinder::getHitSelDynamicWindows(
      const float invpt, const float theta, float &min_dq, float &max_dq, float &min_dphi, float &max_dphi) {
    float max_invpt = std::min(invpt, 10.0f);  // => pT>0.1 GeV

    enum SelWinParameters_e { dp_sf = 0, dp_0, dp_1, dp_2, dq_sf, dq_0, dq_1, dq_2 };
    auto &v = m_iteration_layer_config->get_window_params(m_in_fwd, true);

    if (!v.empty()) {
      // dq hit selection window
      float this_dq = v[dq_sf] * (v[dq_0] * max_invpt + v[dq_1] * theta + v[dq_2]);
      // In case value is below 0 (bad window derivation or other reasons), leave original limits
      if (this_dq > 0.f) {
        min_dq = this_dq;
        max_dq = 2.0f * min_dq;
      }

      // dphi hit selection window
      float this_dphi = v[dp_sf] * (v[dp_0] * max_invpt + v[dp_1] * theta + v[dp_2]);
      // In case value is too low (bad window derivation or other reasons), leave original limits
      if (this_dphi > min_dphi) {
        min_dphi = this_dphi;
        max_dphi = 2.0f * min_dphi;
      }
    }
  }

  //==============================================================================
  // getHitSelDynamicChi2Cut
  //==============================================================================
  // From HitSelectionWindows.h: track-related config on hit selection windows

  inline float MkFinder::getHitSelDynamicChi2Cut(const int itrk, const int ipar) {
    const float minChi2Cut = m_iteration_params->chi2Cut_min;
    const float invpt = m_Par[ipar].At(itrk, 3, 0);
    const float theta = std::abs(m_Par[ipar].At(itrk, 5, 0) - Const::PIOver2);

    float max_invpt = std::min(invpt, 10.0f);  // => pT>0.1 GeV

    enum SelWinParameters_e { c2_sf = 8, c2_0, c2_1, c2_2 };
    auto &v = m_iteration_layer_config->get_window_params(m_in_fwd, true);

    if (!v.empty()) {
      float this_c2 = v[c2_sf] * (v[c2_0] * max_invpt + v[c2_1] * theta + v[c2_2]);
      // In case value is too low (bad window derivation or other reasons), leave original limits
      if (this_c2 > minChi2Cut)
        return this_c2;
    }
    return minChi2Cut;
  }

  //==============================================================================
  // SelectHitIndices
  //==============================================================================

  void MkFinder::selectHitIndices(const LayerOfHits &layer_of_hits, const int N_proc, bool fill_binsearch_only) {
    // bool debug = true;
    using bidx_t = LayerOfHits::bin_index_t;
    using bcnt_t = LayerOfHits::bin_content_t;
    const LayerOfHits &L = layer_of_hits;
    const IterationLayerConfig &ILC = *m_iteration_layer_config;

    const int iI = iP;
    const float nSigmaPhi = 3;
    const float nSigmaZ = 3;
    const float nSigmaR = 3;

    dprintf("LayerOfHits::SelectHitIndices %s layer=%d N_proc=%d\n",
            L.is_barrel() ? "barrel" : "endcap",
            L.layer_id(),
            N_proc);

    float dqv[NN], dphiv[NN], qv[NN], phiv[NN];
    bidx_t qb1v[NN], qb2v[NN], qbv[NN], pb1v[NN], pb2v[NN];

    const auto assignbins = [&](int itrack,
                                float q,
                                float dq,
                                float phi,
                                float dphi,
                                float min_dq,
                                float max_dq,
                                float min_dphi,
                                float max_dphi) {
      dphi = std::clamp(std::abs(dphi), min_dphi, max_dphi);
      dq = std::clamp(dq, min_dq, max_dq);
      //
      qv[itrack] = q;
      phiv[itrack] = phi;
      dphiv[itrack] = dphi;
      dqv[itrack] = dq;
      //
      qbv[itrack] = L.qBinChecked(q);
      qb1v[itrack] = L.qBinChecked(q - dq);
      qb2v[itrack] = L.qBinChecked(q + dq) + 1;
      pb1v[itrack] = L.phiBinChecked(phi - dphi);
      pb2v[itrack] = L.phiMaskApply(L.phiBin(phi + dphi) + 1);
    };

    const auto calcdphi2 = [&](int itrack, float dphidx, float dphidy) {
      return dphidx * dphidx * m_Err[iI].constAt(itrack, 0, 0) + dphidy * dphidy * m_Err[iI].constAt(itrack, 1, 1) +
             2 * dphidx * dphidy * m_Err[iI].constAt(itrack, 0, 1);
    };

    const auto calcdphi = [&](float dphi2, float min_dphi) {
      return std::max(nSigmaPhi * std::sqrt(std::abs(dphi2)), min_dphi);
    };

    if (L.is_barrel()) {
      // Pull out the part of the loop that vectorizes with icc and gcc
      // In llvm16 clang issues a warning that it can't vectorize
      // the loop.  Unfortunately, there doesn't seem to be a
      // pragma to suppress the warning, so we ifdef it out.  This
      // should be rechecked if llvm vectorization improves.
#if !defined(__clang__)
#pragma omp simd
#endif
      for (int itrack = 0; itrack < NN; ++itrack) {
        m_XHitSize[itrack] = 0;

        float min_dq = ILC.min_dq();
        float max_dq = ILC.max_dq();
        float min_dphi = ILC.min_dphi();
        float max_dphi = ILC.max_dphi();

        const float invpt = m_Par[iI].At(itrack, 3, 0);
        const float theta = std::fabs(m_Par[iI].At(itrack, 5, 0) - Const::PIOver2);
        getHitSelDynamicWindows(invpt, theta, min_dq, max_dq, min_dphi, max_dphi);

        const float x = m_Par[iI].constAt(itrack, 0, 0);
        const float y = m_Par[iI].constAt(itrack, 1, 0);
        const float r2 = x * x + y * y;
        const float dphidx = -y / r2, dphidy = x / r2;
        const float dphi2 = calcdphi2(itrack, dphidx, dphidy);
#ifdef HARD_CHECK
        assert(dphi2 >= 0);
#endif

        const float phi = getPhi(x, y);
        float dphi = calcdphi(dphi2, min_dphi);

        const float z = m_Par[iI].constAt(itrack, 2, 0);
        const float dz = std::abs(nSigmaZ * std::sqrt(m_Err[iI].constAt(itrack, 2, 2)));
        const float edgeCorr = std::abs(0.5f * (L.layer_info()->rout() - L.layer_info()->rin()) /
                                        std::tan(m_Par[iI].constAt(itrack, 5, 0)));
        // XXX-NUM-ERR above, m_Err(2,2) gets negative!

        m_XWsrResult[itrack] = L.is_within_z_sensitive_region(z, std::sqrt(dz * dz + edgeCorr * edgeCorr));
        assignbins(itrack, z, dz, phi, dphi, min_dq, max_dq, min_dphi, max_dphi);

        // Relax propagation-fail detection to be in line with pre-43145.
        if (m_FailFlag[itrack] && std::sqrt(r2) >= L.layer_info()->rin()) {
          m_FailFlag[itrack] = 0;
        }
      }
    } else  // endcap
    {
      //layer half-thikness for dphi spread calculation; only for very restrictive iters
      const float layerD = std::abs(L.layer_info()->zmax() - L.layer_info()->zmin()) * 0.5f *
                           (m_iteration_params->maxConsecHoles == 0 || m_iteration_params->maxHolesPerCand == 0);
      // Pull out the part of the loop that vectorizes with icc and gcc
#if !defined(__clang__)
#pragma omp simd
#endif
      for (int itrack = 0; itrack < NN; ++itrack) {
        m_XHitSize[itrack] = 0;

        float min_dq = ILC.min_dq();
        float max_dq = ILC.max_dq();
        float min_dphi = ILC.min_dphi();
        float max_dphi = ILC.max_dphi();

        const float invpt = m_Par[iI].At(itrack, 3, 0);
        const float theta = std::fabs(m_Par[iI].At(itrack, 5, 0) - Const::PIOver2);
        getHitSelDynamicWindows(invpt, theta, min_dq, max_dq, min_dphi, max_dphi);

        const float x = m_Par[iI].constAt(itrack, 0, 0);
        const float y = m_Par[iI].constAt(itrack, 1, 0);
        const float r2 = x * x + y * y;
        const float r2Inv = 1.f / r2;
        const float dphidx = -y * r2Inv, dphidy = x * r2Inv;
        const float phi = getPhi(x, y);
        const float dphi2 =
            calcdphi2(itrack, dphidx, dphidy)
            //range from finite layer thickness
            + std::pow(layerD * std::tan(m_Par[iI].At(itrack, 5, 0)) * std::sin(m_Par[iI].At(itrack, 4, 0) - phi), 2) *
                  r2Inv;
#ifdef HARD_CHECK
        assert(dphi2 >= 0);
#endif

        float dphi = calcdphi(dphi2, min_dphi);

        const float r = std::sqrt(r2);
        const float dr = nSigmaR * std::sqrt(std::abs(x * x * m_Err[iI].constAt(itrack, 0, 0) +
                                                      y * y * m_Err[iI].constAt(itrack, 1, 1) +
                                                      2 * x * y * m_Err[iI].constAt(itrack, 0, 1)) /
                                             r2);
        const float edgeCorr = std::abs(0.5f * (L.layer_info()->zmax() - L.layer_info()->zmin()) *
                                        std::tan(m_Par[iI].constAt(itrack, 5, 0)));

        m_XWsrResult[itrack] = L.is_within_r_sensitive_region(r, std::sqrt(dr * dr + edgeCorr * edgeCorr));
        assignbins(itrack, r, dr, phi, dphi, min_dq, max_dq, min_dphi, max_dphi);
      }
    }

#ifdef RNT_DUMP_MkF_SelHitIdcs
    if (fill_binsearch_only) {
      // XXX loop over good indices (prepared in V2) and put in V1 BinSearch results
      for (auto i : rnt_shi.f_h_idcs) {
        CandInfo &ci = (*rnt_shi.ci)[rnt_shi.f_h_remap[i]];
        ci.bso = BinSearch({phiv[i],
                            dphiv[i],
                            qv[i],
                            dqv[i],
                            pb1v[i],
                            pb2v[i],
                            qb1v[i],
                            qb2v[i],
                            m_XWsrResult[i].m_wsr,
                            m_XWsrResult[i].m_in_gap,
                            false});
      }
      return;
    }
#endif

    // Vectorizing this makes it run slower!
    //#pragma omp simd
    for (int itrack = 0; itrack < N_proc; ++itrack) {
      // PROP-FAIL-ENABLE The following to be enabled when propagation failure
      // detection is properly implemented in propagate-to-R/Z.
      if (m_FailFlag[itrack]) {
        m_XWsrResult[itrack].m_wsr = WSR_Failed;
        continue;
      }

      if (m_XWsrResult[itrack].m_wsr == WSR_Outside) {
        continue;
      }

      const bidx_t qb = qbv[itrack];
      const bidx_t qb1 = qb1v[itrack];
      const bidx_t qb2 = qb2v[itrack];
      const bidx_t pb1 = pb1v[itrack];
      const bidx_t pb2 = pb2v[itrack];

      const float q = qv[itrack];
      const float phi = phiv[itrack];
      const float dphi = dphiv[itrack];
      const float dq = dqv[itrack];

      // clang-format off
      dprintf("  %2d/%2d: %6.3f %6.3f %6.6f %7.5f %3u %3u %4u %4u\n",
              L.layer_id(), itrack, q, phi, dq, dphi,
              qb1, qb2, pb1, pb2);
      // clang-format on

#if defined(DUMPHITWINDOW) && defined(MKFIT_STANDALONE)
      const auto ngr = [](float f) { return isFinite(f) ? f : -999.0f; };

      const int seed_lbl = m_event->currentSeed(m_SeedOriginIdx[itrack]).label();
      Event::SimLabelFromHits slfh = m_event->simLabelForCurrentSeed(m_SeedOriginIdx[itrack]);
      const int seed_mcid = (slfh.is_set() && slfh.good_frac() > 0.7f) ? slfh.label : -999999;
#endif

      for (bidx_t qi = qb1; qi != qb2; ++qi) {
        for (bidx_t pi = pb1; pi != pb2; pi = L.phiMaskApply(pi + 1)) {
          // Limit to central Q-bin
          if (qi == qb && L.isBinDead(pi, qi) == true) {
            dprint("dead module for track in layer=" << L.layer_id() << " qb=" << qi << " pi=" << pi << " q=" << q
                                                     << " phi=" << phi);
            m_XWsrResult[itrack].m_in_gap = true;
          }

          // MT: The following line is the biggest hog (4% total run time).
          // This comes from cache misses, I presume.
          // It might make sense to make first loop to extract bin indices
          // and issue prefetches at the same time.
          // Then enter vectorized loop to actually collect the hits in proper order.

          //SK: ~20x1024 bin sizes give mostly 1 hit per bin. Commented out for 128 bins or less
          // #pragma nounroll
          auto pbi = L.phiQBinContent(pi, qi);
          for (bcnt_t hi = pbi.begin(); hi < pbi.end(); ++hi) {
            // MT: Access into m_hit_zs and m_hit_phis is 1% run-time each.

            const unsigned int hi_orig = L.getOriginalHitIndex(hi);

            if (m_iteration_hit_mask && (*m_iteration_hit_mask)[hi_orig]) {
              dprintf(
                  "Yay, denying masked hit on layer %u, hi %u, orig idx %u\n", L.layer_info()->layer_id(), hi, hi_orig);
              continue;
            }

            if (Config::usePhiQArrays) {
              if (m_XHitSize[itrack] >= MPlexHitIdxMax)
                break;

              const float ddq = std::abs(q - L.hit_q(hi));
              const float ddphi = cdist(std::abs(phi - L.hit_phi(hi)));

              // clang-format off
              dprintf("     SHI %3u %4u %5u  %6.3f %6.3f %6.4f %7.5f   %s\n",
                      qi, pi, hi, L.hit_q(hi), L.hit_phi(hi),
                      ddq, ddphi, (ddq < dq && ddphi < dphi) ? "PASS" : "FAIL");
              // clang-format on

#if defined(DUMPHITWINDOW) && defined(MKFIT_STANDALONE)
              // clang-format off
              MPlexQF thisOutChi2;
              {
                const MCHitInfo &mchinfo = m_event->simHitsInfo_[L.refHit(hi).mcHitID()];
                int mchid = mchinfo.mcTrackID();
                int st_isfindable = 0;
                int st_label = -999999;
                int st_prodtype = 0;
                int st_nhits = -1;
                int st_charge = 0;
                float st_r = -999.;
                float st_z = -999.;
                float st_pt = -999.;
                float st_eta = -999.;
                float st_phi = -999.;
                if (mchid >= 0) {
                  Track simtrack = m_event->simTracks_[mchid];
                  st_isfindable = (int)simtrack.isFindable();
                  st_label = simtrack.label();
                  st_prodtype = (int)simtrack.prodType();
                  st_pt = simtrack.pT();
                  st_eta = simtrack.momEta();
                  st_phi = simtrack.momPhi();
                  st_nhits = simtrack.nTotalHits();
                  st_charge = simtrack.charge();
                  st_r = simtrack.posR();
                  st_z = simtrack.z();
                }

                const Hit &thishit = L.refHit(hi_orig);
                m_msErr.copyIn(itrack, thishit.errArray());
                m_msPar.copyIn(itrack, thishit.posArray());

                MPlexQI propFail;
                MPlexLV tmpPropPar;
                const FindingFoos &fnd_foos = FindingFoos::get_finding_foos(L.is_barrel());
                (*fnd_foos.m_compute_chi2_foo)(m_Err[iI],
                                               m_Par[iI],
                                               m_Chg,
                                               m_msErr,
                                               m_msPar,
                                               thisOutChi2,
                                               tmpPropPar,
                                               propFail,
                                               N_proc,
                                               m_prop_config->finding_intra_layer_pflags,
                                               m_prop_config->finding_requires_propagation_to_hit_pos);

                float hx = thishit.x();
                float hy = thishit.y();
                float hz = thishit.z();
                float hr = std::hypot(hx, hy);
                float hphi = std::atan2(hy, hx);
                float hex = ngr( std::sqrt(thishit.exx()) );
                float hey = ngr( std::sqrt(thishit.eyy()) );
                float hez = ngr( std::sqrt(thishit.ezz()) );
                float her = ngr( std::sqrt(
                    (hx * hx * thishit.exx() + hy * hy * thishit.eyy() + 2.0f * hx * hy * m_msErr.At(itrack, 0, 1)) /
                    (hr * hr)) );
                float hephi = ngr( std::sqrt(thishit.ephi()) );
                float hchi2 = ngr( thisOutChi2[itrack] );
                float tx = m_Par[iI].At(itrack, 0, 0);
                float ty = m_Par[iI].At(itrack, 1, 0);
                float tz = m_Par[iI].At(itrack, 2, 0);
                float tr = std::hypot(tx, ty);
                float tphi = std::atan2(ty, tx);
                // float tchi2 = ngr( m_Chi2(itrack, 0, 0) ); // unused
                float tex = ngr( std::sqrt(m_Err[iI].At(itrack, 0, 0)) );
                float tey = ngr( std::sqrt(m_Err[iI].At(itrack, 1, 1)) );
                float tez = ngr( std::sqrt(m_Err[iI].At(itrack, 2, 2)) );
                float ter = ngr( std::sqrt(
                    (tx * tx * tex * tex + ty * ty * tey * tey + 2.0f * tx * ty * m_Err[iI].At(itrack, 0, 1)) /
                    (tr * tr)) );
                float tephi = ngr( std::sqrt(
                    (ty * ty * tex * tex + tx * tx * tey * tey - 2.0f * tx * ty * m_Err[iI].At(itrack, 0, 1)) /
                    (tr * tr * tr * tr)) );
                float ht_dxy = std::hypot(hx - tx, hy - ty);
                float ht_dz = hz - tz;
                float ht_dphi = cdist(std::abs(hphi - tphi));

                static bool first = true;
                if (first) {
                  printf(
                      "HITWINDOWSEL "
                      "evt_id/I:track_algo/I:"
                      "lyr_id/I:lyr_isbrl/I:hit_idx/I:"
                      "trk_cnt/I:trk_idx/I:trk_label/I:"
                      "trk_pt/F:trk_eta/F:trk_mphi/F:trk_chi2/F:"
                      "nhits/I:"
                      "seed_idx/I:seed_label/I:seed_mcid/I:"
                      "hit_mcid/I:"
                      "st_isfindable/I:st_prodtype/I:st_label/I:"
                      "st_pt/F:st_eta/F:st_phi/F:"
                      "st_nhits/I:st_charge/I:st_r/F:st_z/F:"
                      "trk_q/F:hit_q/F:dq_trkhit/F:dq_cut/F:trk_phi/F:hit_phi/F:dphi_trkhit/F:dphi_cut/F:"
                      "t_x/F:t_y/F:t_r/F:t_phi/F:t_z/F:"
                      "t_ex/F:t_ey/F:t_er/F:t_ephi/F:t_ez/F:"
                      "h_x/F:h_y/F:h_r/F:h_phi/F:h_z/F:"
                      "h_ex/F:h_ey/F:h_er/F:h_ephi/F:h_ez/F:"
                      "ht_dxy/F:ht_dz/F:ht_dphi/F:"
                      "h_chi2/F"
                      "\n");
                  first = false;
                }

                if (!(std::isnan(phi)) && !(std::isnan(getEta(m_Par[iI].At(itrack, 5, 0))))) {
                  //|| std::isnan(ter) || std::isnan(her) || std::isnan(m_Chi2(itrack, 0, 0)) || std::isnan(hchi2)))
                  printf("HITWINDOWSEL "
                         "%d %d"
                         "%d %d %d "
                         "%d %d %d "
                         "%6.3f %6.3f %6.3f %6.3f "
                         "%d "
                         "%d %d %d "
                         "%d "
                         "%d %d %d "
                         "%6.3f %6.3f %6.3f "
                         "%d %d %6.3f %6.3f "
                         "%6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f "
                         "%6.3f %6.3f %6.3f %6.3f %6.3f "
                         "%6.6f %6.6f %6.6f %6.6f %6.6f "
                         "%6.3f %6.3f %6.3f %6.3f %6.3f "
                         "%6.6f %6.6f %6.6f %6.6f %6.6f "
                         "%6.3f %6.3f %6.3f "
                         "%6.3f"
                         "\n",
                         m_event->evtID(), m_iteration_config->m_track_algorithm,
                         L.layer_id(), L.is_barrel(), hi_orig,
                         itrack, m_CandIdx(itrack, 0, 0), m_Label(itrack, 0, 0),
                         1.0f / m_Par[iI].At(itrack, 3, 0), getEta(m_Par[iI].At(itrack, 5, 0)), m_Par[iI].At(itrack, 4, 0), m_Chi2(itrack, 0, 0),
                         m_NFoundHits(itrack, 0, 0),
                         m_SeedOriginIdx(itrack, 0, 0), seed_lbl, seed_mcid,
                         mchid,
                         st_isfindable, st_prodtype, st_label,
                         st_pt, st_eta, st_phi,
                         st_nhits, st_charge, st_r, st_z,
                         q, L.hit_q(hi), ddq, dq, phi, L.hit_phi(hi), ddphi, dphi,
                         tx, ty, tr, tphi, tz,
                         tex, tey, ter, tephi, tez,
                         hx, hy, hr, hphi, hz,
                         hex, hey, her, hephi, hez,
                         ht_dxy, ht_dz, ht_dphi,
                         hchi2);
                }
              }
              // clang-format on
#endif

              if (ddq >= dq)
                continue;
              if (ddphi >= dphi)
                continue;

              // MT: Removing extra check gives full efficiency ...
              //     and means our error estimations are wrong!
              // Avi says we should have *minimal* search windows per layer.
              // Also ... if bins are sufficiently small, we do not need the extra
              // checks, see above.
              m_XHitArr.At(itrack, m_XHitSize[itrack]++, 0) = hi_orig;
            } else {
              // MT: The following check alone makes more sense with spiral traversal,
              // we'd be taking in closest hits first.

              // Hmmh -- there used to be some more checks here.
              // Or, at least, the phi binning was much smaller and no further checks were done.
              assert(false && "this code has not been used in a while -- see comments in code");

              if (m_XHitSize[itrack] < MPlexHitIdxMax) {
                m_XHitArr.At(itrack, m_XHitSize[itrack]++, 0) = hi_orig;
              }
            }
          }  //hi
        }    //pi
      }      //qi
    }        //itrack
  }

  //==============================================================================
  // SelectHitIndicesV2
  //==============================================================================

  void MkFinder::selectHitIndicesV2(const LayerOfHits &layer_of_hits, const int N_proc) {
    // bool debug = true;
    using bidx_t = LayerOfHits::bin_index_t;
    using bcnt_t = LayerOfHits::bin_content_t;
    const LayerOfHits &L = layer_of_hits;
    const LayerInfo &LI = *L.layer_info();

    const int iI = iP;

    dprintf("LayerOfHits::SelectHitIndicesV2 %s layer=%d N_proc=%d\n",
            L.is_barrel() ? "barrel" : "endcap",
            L.layer_id(),
            N_proc);

#ifdef RNT_DUMP_MkF_SelHitIdcs
    rnt_shi.InnerIdcsReset(N_proc);
    for (int i = 0; i < N_proc; ++i) {
      auto slfh = m_event->simLabelForCurrentSeed(m_SeedOriginIdx[i]);
      if (m_FailFlag[i]) {
        rnt_shi.RegisterFailedProp(i, m_Par[1 - iI], m_Par[iI], m_event, m_SeedOriginIdx[i]);
      } else if (slfh.is_set()) {
        rnt_shi.RegisterGoodProp(i, m_Par[iI], m_event, m_SeedOriginIdx[i]);
        // get BinSearch result from V1.
        selectHitIndices(layer_of_hits, N_proc, true);
      }  // else ... could do something about the bad seeds ... probably better to collect elsewhere.
    }
#endif

    constexpr int NEW_MAX_HIT = 6;  // 4 - 6 give about the same # of tracks in quality-val
    constexpr float DDPHI_PRESEL_FAC = 2.0f;
    constexpr float DDQ_PRESEL_FAC = 1.2f;
    constexpr float PHI_BIN_EXTRA_FAC = 2.75f;
    constexpr float Q_BIN_EXTRA_FAC = 1.6f;

    namespace mp = mini_propagators;
    struct Bins {
      MPlexQUH q0, q1, q2, p1, p2;
      mp::InitialStatePlex isp;
      mp::StatePlex sp1, sp2;
      int n_proc;

      MPlexQF dphi_track, dq_track;  // 3 sigma track errors at initial state

      // debug & ntuple dump -- to be local in functions
      MPlexQF phi_c, dphi;
      MPlexQF q_c, qmin, qmax;

      Bins(const MPlexLV &par, const MPlexQI &chg, int np = NN) : isp(par, chg), n_proc(np) {}

      void prop_to_limits(const LayerInfo &li) {
        // Positions 1 and 2 should really be by "propagation order", 1 is the closest/
        // This should also work for backward propagation so not exactly trivial.
        // Also, do not really need propagation to center.
        if (li.is_barrel()) {
          isp.propagate_to_r(mp::PA_Exact, li.rin(), sp1, true, n_proc);
          isp.propagate_to_r(mp::PA_Exact, li.rout(), sp2, true, n_proc);
        } else {
          isp.propagate_to_z(mp::PA_Exact, li.zmin(), sp1, true, n_proc);
          isp.propagate_to_z(mp::PA_Exact, li.zmax(), sp2, true, n_proc);
        }
      }

      void find_bin_ranges(const LayerInfo &li, const LayerOfHits &loh, const MPlexLS &err) {
        // Below made members for debugging
        // MPlexQF phi_c, dphi_min, dphi_max;
        // phi_c = mp::fast_atan2(isp.y, isp.x);  // calculated below as difference

        // Matriplex::min_max(sp1.dphi, sp2.dphi, dphi_min, dphi_max);
        // the above is wrong: dalpha is not dphi --> renamed variable in State
        MPlexQF xp1, xp2, pmin, pmax;
        xp1 = mp::fast_atan2(sp1.y, sp1.x);
        xp2 = mp::fast_atan2(sp2.y, sp2.x);
        Matriplex::min_max(xp1, xp2, pmin, pmax);
        // Matriplex::min_max(mp::fast_atan2(sp1.y, sp1.x), smp::fast_atan2(sp2.y, sp2.x), pmin, pmax);
        MPlexQF dp = pmax - pmin;
        phi_c = 0.5f * (pmax + pmin);
        for (int ii = 0; ii < n_proc; ++ii) {
          if (dp[ii] > Const::PI) {
            std::swap(pmax[ii], pmin[ii]);
            dp[ii] = Const::TwoPI - dp[ii];
            phi_c[ii] = Const::PI - phi_c[ii];
          }
          dphi[ii] = 0.5f * dp[ii];
          // printf("phic: %f  p1: %f  p2: %f   pmin: %f  pmax: %f   dphi: %f\n",
          //       phi_c[ii], xp1[ii], xp2[ii], pmin[ii], pmax[ii], dphi[ii]);
        }

        const auto calc_err_xy = [&](const MPlexQF &x, const MPlexQF &y) {
          return x * x * err.ReduceFixedIJ(0, 0) + y * y * err.ReduceFixedIJ(1, 1) +
                 2.0f * x * y * err.ReduceFixedIJ(0, 1);
        };

        // Calculate dphi_track, dq_track differs for barrel/endcap
        MPlexQF r2_c = isp.x * isp.x + isp.y * isp.y;
        MPlexQF r2inv_c = 1.0f / r2_c;
        MPlexQF dphidx_c = -isp.y * r2inv_c;
        MPlexQF dphidy_c = isp.x * r2inv_c;
        dphi_track = 3.0f * calc_err_xy(dphidx_c, dphidy_c).abs().sqrt();

        // MPlexQF qmin, qmax;
        if (li.is_barrel()) {
          Matriplex::min_max(sp1.z, sp2.z, qmin, qmax);
          q_c = isp.z;
          dq_track = 3.0f * err.ReduceFixedIJ(2, 2).abs().sqrt();
        } else {
          Matriplex::min_max(Matriplex::hypot(sp1.x, sp1.y), Matriplex::hypot(sp2.x, sp2.y), qmin, qmax);
          q_c = Matriplex::sqrt(r2_c);
          dq_track = 3.0f * (r2inv_c * calc_err_xy(isp.x, isp.y).abs()).sqrt();
        }

        for (int i = 0; i < p1.kTotSize; ++i) {
          // Clamp crazy sizes. This actually only happens when prop-fail flag is set.
          // const float dphi_clamp = 0.1;
          // if (dphi_min[i] > 0.0f || dphi_min[i] < -dphi_clamp) dphi_min[i] = -dphi_clamp;
          // if (dphi_max[i] < 0.0f || dphi_max[i] > dphi_clampf) dphi_max[i] = dphi_clamp;
          p1[i] = loh.phiBinChecked(pmin[i] - dphi_track[i] - PHI_BIN_EXTRA_FAC * 0.0123f);
          p2[i] = loh.phiBinChecked(pmax[i] + dphi_track[i] + PHI_BIN_EXTRA_FAC * 0.0123f);

          q0[i] = loh.qBinChecked(q_c[i]);
          q1[i] = loh.qBinChecked(qmin[i] - dq_track[i] - Q_BIN_EXTRA_FAC * 0.5f * li.q_bin());
          q2[i] = loh.qBinChecked(qmax[i] + dq_track[i] + Q_BIN_EXTRA_FAC * 0.5f * li.q_bin()) + 1;
        }
      }
    };

    Bins B(m_Par[iI], m_Chg, N_proc);
    B.prop_to_limits(LI);
    B.find_bin_ranges(LI, L, m_Err[iI]);

    for (int i = 0; i < N_proc; ++i) {
      m_XHitSize[i] = 0;
      // Notify failure. Ideally should be detected before selectHitIndices().
      if (m_FailFlag[i]) {
        m_XWsrResult[i].m_wsr = WSR_Failed;
      } else {
        if (LI.is_barrel()) {
          m_XWsrResult[i] = L.is_within_z_sensitive_region(B.q_c[i], 0.5f * (B.q2[i] - B.q1[i]));
        } else {
          m_XWsrResult[i] = L.is_within_r_sensitive_region(B.q_c[i], 0.5f * (B.q2[i] - B.q1[i]));
        }
      }
    }

    // for (int i = 0; i < N_proc; ++i) {
    //   printf("BinCheck %c %+8.6f %+8.6f | %3d %3d - %3d %3d ||  | %2d %2d - %2d %2d\n", LI.is_barrel() ? 'B' : 'E',
    //          B.phi[i], B.dphi[i], B.p1[i], B.p2[i], pb1v[i], pb2v[i],
    //          B.q[i], B.dq[i], B.q1[i], B.q2[i], qb1v[i], qb2v[i]);
    // }

#ifdef RNT_DUMP_MkF_SelHitIdcs
    for (auto i : rnt_shi.f_h_idcs) {
      CandInfo &ci = (*rnt_shi.ci)[rnt_shi.f_h_remap[i]];
      ci.bsn = BinSearch({B.phi_c[i],
                          B.dphi[i],
                          B.q_c[i],
                          0.5f * (B.q2[i] - B.q1[i]),
                          B.p1[i],
                          B.p2[i],
                          B.q1[i],
                          B.q2[i],
                          m_XWsrResult[i].m_wsr,
                          m_XWsrResult[i].m_in_gap,
                          false});
      ci.ps_min = statep2propstate(B.sp1, i);
      ci.ps_max = statep2propstate(B.sp2, i);
    }
#endif

    struct PQE {
      float score;
      unsigned int hit_index;
    };
    auto pqe_cmp = [](const PQE &a, const PQE &b) { return a.score < b.score; };
    std::priority_queue<PQE, std::vector<PQE>, decltype(pqe_cmp)> pqueue(pqe_cmp);
    int pqueue_size = 0;

    // Vectorizing this makes it run slower!
    //#pragma omp simd
    for (int itrack = 0; itrack < N_proc; ++itrack) {
      if (m_FailFlag[itrack]) {
        m_XWsrResult[itrack].m_wsr = WSR_Failed;
        continue;
      }

      if (m_XWsrResult[itrack].m_wsr == WSR_Outside) {
        continue;
      }

      // New binning -- known to be too restrictive, so scaled up. Probably esp. in stereo layers.
      // Also, could take track covariance dphi / dq extras + known tilt stuff.
      const bidx_t qb = B.q0[itrack];
      const bidx_t qb1 = B.q1[itrack];
      const bidx_t qb2 = B.q2[itrack];
      const bidx_t pb1 = B.p1[itrack];
      const bidx_t pb2 = B.p2[itrack];

      // clang-format off
      dprintf("  %2d/%2d: %6.3f %6.3f %6.6f %7.5f %3u %3u %4u %4u\n",
              L.layer_id(), itrack, qv[itrack], phi[itrack], dqv[itrack], dphiv[itrack],
              qb1, qb2, pb1, pb2);
      // clang-format on

      mp::InitialState mp_is(m_Par[iI], m_Chg, itrack);
      mp::State mp_s;

      for (bidx_t qi = qb1; qi != qb2; ++qi) {
        for (bidx_t pi = pb1; pi != pb2; pi = L.phiMaskApply(pi + 1)) {
          // Limit to central Q-bin
          if (qi == qb && L.isBinDead(pi, qi) == true) {
            dprint("dead module for track in layer=" << L.layer_id() << " qb=" << qi << " pi=" << pi << " q=" << q
                                                     << " phi=" << phi);
            m_XWsrResult[itrack].m_in_gap = true;
          }

          // It might make sense to make first loop to extract bin indices
          // and issue prefetches at the same time.
          // Then enter vectorized loop to actually collect the hits in proper order.

          //SK: ~20x1024 bin sizes give mostly 1 hit per bin. Commented out for 128 bins or less
          // #pragma nounroll
          auto pbi = L.phiQBinContent(pi, qi);
          for (bcnt_t hi = pbi.begin(); hi < pbi.end(); ++hi) {
            // MT: Access into m_hit_zs and m_hit_phis is 1% run-time each.

            const unsigned int hi_orig = L.getOriginalHitIndex(hi);

            if (m_iteration_hit_mask && (*m_iteration_hit_mask)[hi_orig]) {
              dprintf(
                  "Yay, denying masked hit on layer %u, hi %u, orig idx %u\n", L.layer_info()->layer_id(), hi, hi_orig);
              continue;
            }

            if (m_XHitSize[itrack] >= MPlexHitIdxMax)
              break;

            float new_q, new_phi, new_ddphi, new_ddq;
            bool prop_fail;

            if (L.is_barrel()) {
              prop_fail = mp_is.propagate_to_r(mp::PA_Exact, L.hit_qbar(hi), mp_s, true);
              new_q = mp_s.z;
            } else {
              prop_fail = mp_is.propagate_to_z(mp::PA_Exact, L.hit_qbar(hi), mp_s, true);
              new_q = std::hypot(mp_s.x, mp_s.y);
            }

            new_phi = vdt::fast_atan2f(mp_s.y, mp_s.x);
            new_ddphi = cdist(std::abs(new_phi - L.hit_phi(hi)));
            new_ddq = std::abs(new_q - L.hit_q(hi));

            bool dqdphi_presel = new_ddq < B.dq_track[itrack] + DDQ_PRESEL_FAC * L.hit_q_half_length(hi) &&
                                 new_ddphi < B.dphi_track[itrack] + DDPHI_PRESEL_FAC * 0.0123f;

            // clang-format off
            dprintf("     SHI %3u %4u %5u  %6.3f %6.3f %6.4f %7.5f  PROP-%s  %s\n",
                    qi, pi, hi, L.hit_q(hi), L.hit_phi(hi),
                    ddq, ddphi, prop_fail ? "FAIL" : "OK", dqdphi_presel ? "PASS" : "REJECT");
            // clang-format on

            if (prop_fail || !dqdphi_presel)
              continue;
            if (pqueue_size < NEW_MAX_HIT) {
              pqueue.push({new_ddphi, hi_orig});
              ++pqueue_size;
            } else if (new_ddphi < pqueue.top().score) {
              pqueue.pop();
              pqueue.push({new_ddphi, hi_orig});
            }
          }  //hi
        }    //pi
      }      //qi

      dprintf(" PQUEUE (%d)", pqueue_size);
      // Reverse hits so best dphis/scores come first in the hit-index list.
      m_XHitSize[itrack] = pqueue_size;
      while (pqueue_size) {
        --pqueue_size;
        m_XHitArr.At(itrack, pqueue_size, 0) = pqueue.top().hit_index;
        dprintf("   %d: %f %d", pqueue_size, pqueue.top().score, pqueue.top().hit_index);
        pqueue.pop();
      }
      dprintf("\n");

    }  //itrack
  }

  //==============================================================================
  // AddBestHit - Best Hit Track Finding
  //==============================================================================

  void MkFinder::addBestHit(const LayerOfHits &layer_of_hits, const int N_proc, const FindingFoos &fnd_foos) {
    // debug = true;

    MatriplexHitPacker mhp(layer_of_hits.hitArray());

    float minChi2[NN];
    int bestHit[NN];
    int maxSize = 0;

    // Determine maximum number of hits for tracks in the collection.
    for (int it = 0; it < NN; ++it) {
      if (it < N_proc) {
        if (m_XHitSize[it] > 0) {
          maxSize = std::max(maxSize, m_XHitSize[it]);
        }
      }

      bestHit[it] = -1;
      minChi2[it] = getHitSelDynamicChi2Cut(it, iP);
    }

    for (int hit_cnt = 0; hit_cnt < maxSize; ++hit_cnt) {
      //fixme what if size is zero???

      mhp.reset();

      //#pragma omp simd doesn't vectorize with current compilers
      for (int itrack = 0; itrack < N_proc; ++itrack) {
        if (hit_cnt < m_XHitSize[itrack]) {
          mhp.addInputAt(itrack, layer_of_hits.refHit(m_XHitArr.At(itrack, hit_cnt, 0)));
        }
      }

      mhp.pack(m_msErr, m_msPar);

      //now compute the chi2 of track state vs hit
      MPlexQF outChi2;
      MPlexLV tmpPropPar;
      clearFailFlag();
      (*fnd_foos.m_compute_chi2_foo)(m_Err[iP],
                                     m_Par[iP],
                                     m_Chg,
                                     m_msErr,
                                     m_msPar,
                                     outChi2,
                                     tmpPropPar,
                                     m_FailFlag,
                                     N_proc,
                                     m_prop_config->finding_intra_layer_pflags,
                                     m_prop_config->finding_requires_propagation_to_hit_pos);

      //update best hit in case chi2<minChi2
#pragma omp simd
      for (int itrack = 0; itrack < N_proc; ++itrack) {
        if (hit_cnt < m_XHitSize[itrack]) {
          const float chi2 = std::abs(outChi2[itrack]);  //fixme negative chi2 sometimes...
          dprint("chi2=" << chi2 << " minChi2[itrack]=" << minChi2[itrack]);
          if (chi2 < minChi2[itrack]) {
            minChi2[itrack] = chi2;
            bestHit[itrack] = m_XHitArr.At(itrack, hit_cnt, 0);
          }
        }
      }
    }  // end loop over hits

    //#pragma omp simd
    for (int itrack = 0; itrack < N_proc; ++itrack) {
      if (m_XWsrResult[itrack].m_wsr == WSR_Outside) {
        // Why am I doing this?
        m_msErr.setDiagonal3x3(itrack, 666);
        m_msPar(itrack, 0, 0) = m_Par[iP](itrack, 0, 0);
        m_msPar(itrack, 1, 0) = m_Par[iP](itrack, 1, 0);
        m_msPar(itrack, 2, 0) = m_Par[iP](itrack, 2, 0);

        // XXXX If not in gap, should get back the old track params. But they are gone ...
        // Would actually have to do it right after SelectHitIndices where updated params are still ok.
        // Here they got screwed during hit matching.
        // So, I'd store them there (into propagated params) and retrieve them here.
        // Or we decide not to care ...

        continue;
      }

      //fixme decide what to do in case no hit found
      if (bestHit[itrack] >= 0) {
        const Hit &hit = layer_of_hits.refHit(bestHit[itrack]);
        const float chi2 = minChi2[itrack];

        dprint("ADD BEST HIT FOR TRACK #"
               << itrack << std::endl
               << "prop x=" << m_Par[iP].constAt(itrack, 0, 0) << " y=" << m_Par[iP].constAt(itrack, 1, 0) << std::endl
               << "copy in hit #" << bestHit[itrack] << " x=" << hit.position()[0] << " y=" << hit.position()[1]);

        m_msErr.copyIn(itrack, hit.errArray());
        m_msPar.copyIn(itrack, hit.posArray());
        m_Chi2(itrack, 0, 0) += chi2;

        add_hit(itrack, bestHit[itrack], layer_of_hits.layer_id());
      } else {
        int fake_hit_idx = Hit::kHitMissIdx;

        if (m_XWsrResult[itrack].m_wsr == WSR_Edge) {
          // YYYYYY Config::store_missed_layers
          fake_hit_idx = Hit::kHitEdgeIdx;
        } else if (num_all_minus_one_hits(itrack)) {
          fake_hit_idx = Hit::kHitStopIdx;
        }

        dprint("ADD FAKE HIT FOR TRACK #" << itrack << " withinBounds=" << (fake_hit_idx != Hit::kHitEdgeIdx)
                                          << " r=" << std::hypot(m_Par[iP](itrack, 0, 0), m_Par[iP](itrack, 1, 0)));

        m_msErr.setDiagonal3x3(itrack, 666);
        m_msPar(itrack, 0, 0) = m_Par[iP](itrack, 0, 0);
        m_msPar(itrack, 1, 0) = m_Par[iP](itrack, 1, 0);
        m_msPar(itrack, 2, 0) = m_Par[iP](itrack, 2, 0);
        // Don't update chi2

        add_hit(itrack, fake_hit_idx, layer_of_hits.layer_id());
      }
    }

    // Update the track parameters with this hit. (Note that some calculations
    // are already done when computing chi2. Not sure it's worth caching them?)

    dprint("update parameters");
    clearFailFlag();
    (*fnd_foos.m_update_param_foo)(m_Err[iP],
                                   m_Par[iP],
                                   m_Chg,
                                   m_msErr,
                                   m_msPar,
                                   m_Err[iC],
                                   m_Par[iC],
                                   m_FailFlag,
                                   N_proc,
                                   m_prop_config->finding_intra_layer_pflags,
                                   m_prop_config->finding_requires_propagation_to_hit_pos);

    dprint("m_Par[iP](0,0,0)=" << m_Par[iP](0, 0, 0) << " m_Par[iC](0,0,0)=" << m_Par[iC](0, 0, 0));
  }

  //=======================================================
  // isStripQCompatible : check if prop is consistent with the barrel/endcap strip length
  //=======================================================
  bool isStripQCompatible(
      int itrack, bool isBarrel, const MPlexLS &pErr, const MPlexLV &pPar, const MPlexHS &msErr, const MPlexHV &msPar) {
    //check module compatibility via long strip side = L/sqrt(12)
    if (isBarrel) {  //check z direction only
      const float res = std::abs(msPar.constAt(itrack, 2, 0) - pPar.constAt(itrack, 2, 0));
      const float hitHL = sqrt(msErr.constAt(itrack, 2, 2) * 3.f);  //half-length
      const float qErr = sqrt(pErr.constAt(itrack, 2, 2));
      dprint("qCompat " << hitHL << " + " << 3.f * qErr << " vs " << res);
      return hitHL + std::max(3.f * qErr, 0.5f) > res;
    } else {  //project on xy, assuming the strip Length >> Width
      const float res[2]{msPar.constAt(itrack, 0, 0) - pPar.constAt(itrack, 0, 0),
                         msPar.constAt(itrack, 1, 0) - pPar.constAt(itrack, 1, 0)};
      const float hitT2 = msErr.constAt(itrack, 0, 0) + msErr.constAt(itrack, 1, 1);
      const float hitT2inv = 1.f / hitT2;
      const float proj[3] = {msErr.constAt(itrack, 0, 0) * hitT2inv,
                             msErr.constAt(itrack, 0, 1) * hitT2inv,
                             msErr.constAt(itrack, 1, 1) * hitT2inv};
      const float qErr =
          sqrt(std::abs(pErr.constAt(itrack, 0, 0) * proj[0] + 2.f * pErr.constAt(itrack, 0, 1) * proj[1] +
                        pErr.constAt(itrack, 1, 1) * proj[2]));  //take abs to avoid non-pos-def cases
      const float resProj =
          sqrt(res[0] * proj[0] * res[0] + 2.f * res[1] * proj[1] * res[0] + res[1] * proj[2] * res[1]);
      dprint("qCompat " << sqrt(hitT2 * 3.f) << " + " << 3.f * qErr << " vs " << resProj);
      return sqrt(hitT2 * 3.f) + std::max(3.f * qErr, 0.5f) > resProj;
    }
  }

  //=======================================================
  // passStripChargePCMfromTrack : apply the slope correction to charge per cm and cut using hit err matrix
  //         the raw pcm = charge/L_normal
  //         the corrected qCorr = charge/L_path = charge/(L_normal*p/p_zLocal) = pcm*p_zLocal/p
  //=======================================================
  bool passStripChargePCMfromTrack(
      int itrack, bool isBarrel, unsigned int pcm, unsigned int pcmMin, const MPlexLV &pPar, const MPlexHS &msErr) {
    //skip the overflow case
    if (pcm >= Hit::maxChargePerCM())
      return true;

    float qSF;
    if (isBarrel) {  //project in x,y, assuming zero-error direction is in this plane
      const float hitT2 = msErr.constAt(itrack, 0, 0) + msErr.constAt(itrack, 1, 1);
      const float hitT2inv = 1.f / hitT2;
      const float proj[3] = {msErr.constAt(itrack, 0, 0) * hitT2inv,
                             msErr.constAt(itrack, 0, 1) * hitT2inv,
                             msErr.constAt(itrack, 1, 1) * hitT2inv};
      const bool detXY_OK =
          std::abs(proj[0] * proj[2] - proj[1] * proj[1]) < 0.1f;  //check that zero-direction is close
      const float cosP = cos(pPar.constAt(itrack, 4, 0));
      const float sinP = sin(pPar.constAt(itrack, 4, 0));
      const float sinT = std::abs(sin(pPar.constAt(itrack, 5, 0)));
      //qSF = sqrt[(px,py)*(1-proj)*(px,py)]/p = sinT*sqrt[(cosP,sinP)*(1-proj)*(cosP,sinP)].
      qSF = detXY_OK ? sinT * std::sqrt(std::abs(1.f + cosP * cosP * proj[0] + sinP * sinP * proj[2] -
                                                 2.f * cosP * sinP * proj[1]))
                     : 1.f;
    } else {  //project on z
      // p_zLocal/p = p_z/p = cosT
      qSF = std::abs(cos(pPar.constAt(itrack, 5, 0)));
    }

    const float qCorr = pcm * qSF;
    dprint("pcm " << pcm << " * " << qSF << " = " << qCorr << " vs " << pcmMin);
    return qCorr > pcmMin;
  }

  //==============================================================================
  // FindCandidates - Standard Track Finding
  //==============================================================================

  void MkFinder::findCandidates(const LayerOfHits &layer_of_hits,
                                std::vector<std::vector<TrackCand>> &tmp_candidates,
                                const int offset,
                                const int N_proc,
                                const FindingFoos &fnd_foos) {
    // bool debug = true;

    MatriplexHitPacker mhp(layer_of_hits.hitArray());

    int maxSize = 0;

    // Determine maximum number of hits for tracks in the collection.
    for (int it = 0; it < NN; ++it) {
      if (it < N_proc) {
        if (m_XHitSize[it] > 0) {
          maxSize = std::max(maxSize, m_XHitSize[it]);
        }
      }
    }

    dprintf("FindCandidates max hits to process=%d\n", maxSize);

    int nHitsAdded[NN]{};
    bool isTooLargeCluster[NN]{false};

    for (int hit_cnt = 0; hit_cnt < maxSize; ++hit_cnt) {
      mhp.reset();

      int charge_pcm[NN];

      //#pragma omp simd doesn't vectorize with current compilers
      for (int itrack = 0; itrack < N_proc; ++itrack) {
        if (hit_cnt < m_XHitSize[itrack]) {
          const auto &hit = layer_of_hits.refHit(m_XHitArr.At(itrack, hit_cnt, 0));
          mhp.addInputAt(itrack, hit);
          charge_pcm[itrack] = hit.chargePerCM();
        }
      }

      mhp.pack(m_msErr, m_msPar);

      //now compute the chi2 of track state vs hit
      MPlexQF outChi2;
      MPlexLV propPar;
      clearFailFlag();

      if (Config::usePropToPlane) {
        // Maybe could use 2 matriplex packers ... ModuleInfo has 3 * SVector3 and uint
        MPlexHV norm, dir;
        packModuleNormDir(layer_of_hits, hit_cnt, norm, dir, N_proc);
        kalmanPropagateAndComputeChi2Plane(m_Err[iP],
                                           m_Par[iP],
                                           m_Chg,
                                           m_msErr,
                                           m_msPar,
                                           norm,
                                           dir,
                                           outChi2,
                                           propPar,
                                           m_FailFlag,
                                           N_proc,
                                           m_prop_config->finding_intra_layer_pflags,
                                           m_prop_config->finding_requires_propagation_to_hit_pos);
      } else {
        (*fnd_foos.m_compute_chi2_foo)(m_Err[iP],
                                       m_Par[iP],
                                       m_Chg,
                                       m_msErr,
                                       m_msPar,
                                       outChi2,
                                       propPar,
                                       m_FailFlag,
                                       N_proc,
                                       m_prop_config->finding_intra_layer_pflags,
                                       m_prop_config->finding_requires_propagation_to_hit_pos);
      }

      // Now update the track parameters with this hit (note that some
      // calculations are already done when computing chi2, to be optimized).
      // 1. This is not needed for candidates the hit is not added to, but it's
      // vectorized so doing it serially below should take the same time.
      // 2. Still it's a waste of time in case the hit is not added to any of the
      // candidates, so check beforehand that at least one cand needs update.
      bool oneCandPassCut = false;
      for (int itrack = 0; itrack < N_proc; ++itrack) {
        float max_c2 = getHitSelDynamicChi2Cut(itrack, iP);

        if (hit_cnt < m_XHitSize[itrack]) {
          const float chi2 = std::abs(outChi2[itrack]);  //fixme negative chi2 sometimes...
          dprint("chi2=" << chi2);
          if (chi2 < max_c2) {
            bool isCompatible = true;
            if (!layer_of_hits.is_pixel()) {
              //check module compatibility via long strip side = L/sqrt(12)
              isCompatible =
                  isStripQCompatible(itrack, layer_of_hits.is_barrel(), m_Err[iP], propPar, m_msErr, m_msPar);

              //rescale strip charge to track parameters and reapply the cut
              isCompatible &= passStripChargePCMfromTrack(
                  itrack, layer_of_hits.is_barrel(), charge_pcm[itrack], Hit::minChargePerCM(), propPar, m_msErr);
            }
            // Select only SiStrip hits with cluster size < maxClusterSize
            if (!layer_of_hits.is_pixel()) {
              if (layer_of_hits.refHit(m_XHitArr.At(itrack, hit_cnt, 0)).spanRows() >=
                  m_iteration_params->maxClusterSize) {
                isTooLargeCluster[itrack] = true;
                isCompatible = false;
              }
            }

            if (isCompatible) {
              oneCandPassCut = true;
              break;
            }
          }
        }
      }

      if (oneCandPassCut) {
        MPlexQI tmpChg = m_Chg;
        clearFailFlag();
        (*fnd_foos.m_update_param_foo)(m_Err[iP],
                                       m_Par[iP],
                                       tmpChg,
                                       m_msErr,
                                       m_msPar,
                                       m_Err[iC],
                                       m_Par[iC],
                                       m_FailFlag,
                                       N_proc,
                                       m_prop_config->finding_intra_layer_pflags,
                                       m_prop_config->finding_requires_propagation_to_hit_pos);

        dprint("update parameters" << std::endl
                                   << "propagated track parameters x=" << m_Par[iP].constAt(0, 0, 0)
                                   << " y=" << m_Par[iP].constAt(0, 1, 0) << std::endl
                                   << "               hit position x=" << m_msPar.constAt(0, 0, 0)
                                   << " y=" << m_msPar.constAt(0, 1, 0) << std::endl
                                   << "   updated track parameters x=" << m_Par[iC].constAt(0, 0, 0)
                                   << " y=" << m_Par[iC].constAt(0, 1, 0));

        //create candidate with hit in case chi2 < max_c2
        //fixme: please vectorize me... (not sure it's possible in this case)
        for (int itrack = 0; itrack < N_proc; ++itrack) {
          float max_c2 = getHitSelDynamicChi2Cut(itrack, iP);

          if (hit_cnt < m_XHitSize[itrack]) {
            const float chi2 = std::abs(outChi2[itrack]);  //fixme negative chi2 sometimes...
            dprint("chi2=" << chi2);
            if (chi2 < max_c2) {
              bool isCompatible = true;
              if (!layer_of_hits.is_pixel()) {
                //check module compatibility via long strip side = L/sqrt(12)
                isCompatible =
                    isStripQCompatible(itrack, layer_of_hits.is_barrel(), m_Err[iP], propPar, m_msErr, m_msPar);

                //rescale strip charge to track parameters and reapply the cut
                isCompatible &= passStripChargePCMfromTrack(
                    itrack, layer_of_hits.is_barrel(), charge_pcm[itrack], Hit::minChargePerCM(), propPar, m_msErr);
              }
              // Select only SiStrip hits with cluster size < maxClusterSize
              if (!layer_of_hits.is_pixel()) {
                if (layer_of_hits.refHit(m_XHitArr.At(itrack, hit_cnt, 0)).spanRows() >=
                    m_iteration_params->maxClusterSize)
                  isCompatible = false;
              }

              if (isCompatible) {
                bool hitExists = false;
                int maxHits = m_NFoundHits(itrack, 0, 0);
                if (layer_of_hits.is_pixel()) {
                  for (int i = 0; i <= maxHits; ++i) {
                    if (i > 2)
                      break;
                    if (m_HoTArrs[itrack][i].layer == layer_of_hits.layer_id()) {
                      hitExists = true;
                      break;
                    }
                  }
                }
                if (hitExists)
                  continue;

                nHitsAdded[itrack]++;
                dprint("chi2 cut passed, creating new candidate");
                // Create a new candidate and fill the tmp_candidates output vector.
                // QQQ only instantiate if it will pass, be better than N_best

                const int hit_idx = m_XHitArr.At(itrack, hit_cnt, 0);

                TrackCand newcand;
                copy_out(newcand, itrack, iC);
                newcand.setCharge(tmpChg(itrack, 0, 0));
                newcand.addHitIdx(hit_idx, layer_of_hits.layer_id(), chi2);
                newcand.setScore(getScoreCand(m_steering_params->m_track_scorer,
                                              newcand,
                                              true /*penalizeTailMissHits*/,
                                              true /*inFindCandidates*/));
                newcand.setOriginIndex(m_CandIdx(itrack, 0, 0));

                // To apply a fixed cut instead of dynamic cut for overlap: m_iteration_params->chi2CutOverlap
                if (chi2 < max_c2) {
                  CombCandidate &ccand = *newcand.combCandidate();
                  ccand[m_CandIdx(itrack, 0, 0)].considerHitForOverlap(
                      hit_idx, layer_of_hits.refHit(hit_idx).detIDinLayer(), chi2);
                }

                dprint("updated track parameters x=" << newcand.parameters()[0] << " y=" << newcand.parameters()[1]
                                                     << " z=" << newcand.parameters()[2]
                                                     << " pt=" << 1. / newcand.parameters()[3]);

                tmp_candidates[m_SeedIdx(itrack, 0, 0) - offset].emplace_back(newcand);
              }
            }
          }
        }
      }  //end if (oneCandPassCut)

    }  //end loop over hits

    //now add invalid hit
    //fixme: please vectorize me...
    for (int itrack = 0; itrack < N_proc; ++itrack) {
      // Cands that miss the layer are stashed away in MkBuilder(), before propagation,
      // and then merged back afterwards.
      if (m_XWsrResult[itrack].m_wsr == WSR_Outside) {
        continue;
      }

      int fake_hit_idx = ((num_all_minus_one_hits(itrack) < m_iteration_params->maxHolesPerCand) &&
                          (m_NTailMinusOneHits(itrack, 0, 0) < m_iteration_params->maxConsecHoles))
                             ? Hit::kHitMissIdx
                             : Hit::kHitStopIdx;

      if (m_XWsrResult[itrack].m_wsr == WSR_Edge) {
        // YYYYYY m_iteration_params->store_missed_layers
        fake_hit_idx = Hit::kHitEdgeIdx;
      }
      //now add fake hit for tracks that passsed through inactive modules
      else if (m_XWsrResult[itrack].m_in_gap == true && nHitsAdded[itrack] == 0) {
        fake_hit_idx = Hit::kHitInGapIdx;
      }
      //now add fake hit for cases where hit cluster size is larger than maxClusterSize
      else if (isTooLargeCluster[itrack] == true && nHitsAdded[itrack] == 0) {
        fake_hit_idx = Hit::kHitMaxClusterIdx;
      }

      dprint("ADD FAKE HIT FOR TRACK #" << itrack << " withinBounds=" << (fake_hit_idx != Hit::kHitEdgeIdx)
                                        << " r=" << std::hypot(m_Par[iP](itrack, 0, 0), m_Par[iP](itrack, 1, 0)));

      // QQQ as above, only create and add if score better
      TrackCand newcand;
      copy_out(newcand, itrack, iP);
      newcand.addHitIdx(fake_hit_idx, layer_of_hits.layer_id(), 0.);
      newcand.setScore(getScoreCand(
          m_steering_params->m_track_scorer, newcand, true /*penalizeTailMissHits*/, true /*inFindCandidates*/));
      // Only relevant when we actually add a hit
      // newcand.setOriginIndex(m_CandIdx(itrack, 0, 0));
      tmp_candidates[m_SeedIdx(itrack, 0, 0) - offset].emplace_back(newcand);
    }
  }

  //==============================================================================
  // FindCandidatesCloneEngine - Clone Engine Track Finding
  //==============================================================================

  void MkFinder::findCandidatesCloneEngine(const LayerOfHits &layer_of_hits,
                                           CandCloner &cloner,
                                           const int offset,
                                           const int N_proc,
                                           const FindingFoos &fnd_foos) {
    // bool debug = true;

    MatriplexHitPacker mhp(layer_of_hits.hitArray());

    int maxSize = 0;

    // Determine maximum number of hits for tracks in the collection.
#pragma omp simd
    for (int it = 0; it < NN; ++it) {
      if (it < N_proc) {
        if (m_XHitSize[it] > 0) {
          maxSize = std::max(maxSize, m_XHitSize[it]);
        }
      }
    }

    dprintf("FindCandidatesCloneEngine max hits to process=%d\n", maxSize);

    int nHitsAdded[NN]{};
    bool isTooLargeCluster[NN]{false};

    for (int hit_cnt = 0; hit_cnt < maxSize; ++hit_cnt) {
      mhp.reset();

      int charge_pcm[NN];

      //#pragma omp simd doesn't vectorize with current compilers
      for (int itrack = 0; itrack < N_proc; ++itrack) {
        if (hit_cnt < m_XHitSize[itrack]) {
          const auto &hit = layer_of_hits.refHit(m_XHitArr.At(itrack, hit_cnt, 0));
          mhp.addInputAt(itrack, hit);
          charge_pcm[itrack] = hit.chargePerCM();
        }
      }

      mhp.pack(m_msErr, m_msPar);

      //now compute the chi2 of track state vs hit
      MPlexQF outChi2;
      MPlexLV propPar;
      clearFailFlag();

      if (Config::usePropToPlane) {
        // Maybe could use 2 matriplex packers ... ModuleInfo has 3 * SVector3 and uint
        MPlexHV norm, dir;
        packModuleNormDir(layer_of_hits, hit_cnt, norm, dir, N_proc);
        kalmanPropagateAndComputeChi2Plane(m_Err[iP],
                                           m_Par[iP],
                                           m_Chg,
                                           m_msErr,
                                           m_msPar,
                                           norm,
                                           dir,
                                           outChi2,
                                           propPar,
                                           m_FailFlag,
                                           N_proc,
                                           m_prop_config->finding_intra_layer_pflags,
                                           m_prop_config->finding_requires_propagation_to_hit_pos);
      } else {
        (*fnd_foos.m_compute_chi2_foo)(m_Err[iP],
                                       m_Par[iP],
                                       m_Chg,
                                       m_msErr,
                                       m_msPar,
                                       outChi2,
                                       propPar,
                                       m_FailFlag,
                                       N_proc,
                                       m_prop_config->finding_intra_layer_pflags,
                                       m_prop_config->finding_requires_propagation_to_hit_pos);
      }

      //#pragma omp simd  // DOES NOT VECTORIZE AS IT IS NOW
      for (int itrack = 0; itrack < N_proc; ++itrack) {
        // We can be in failed state from the initial propagation before selectHitIndices
        // and there hit_count for track is set to -1 and WSR state to Failed, handled below.
        // Or we might have hit it here in propagate-to-hit.
        // PROP-FAIL-ENABLE FailFlag check to be enabled when propagation failure
        // detection is properly implemented in propagate-to-R/Z.
        if (/*!m_FailFlag[itrack] &&*/ hit_cnt < m_XHitSize[itrack]) {
          // make sure the hit was in the compatiblity window for the candidate
          const float max_c2 = getHitSelDynamicChi2Cut(itrack, iP);
          const float chi2 = std::abs(outChi2[itrack]);  //fixme negative chi2 sometimes...
          // XXX-NUM-ERR assert(chi2 >= 0);

          dprint("chi2=" << chi2 << " for trkIdx=" << itrack << " hitIdx=" << m_XHitArr.At(itrack, hit_cnt, 0));
          if (chi2 < max_c2) {
            bool isCompatible = true;
            if (!layer_of_hits.is_pixel()) {
              //check module compatibility via long strip side = L/sqrt(12)
              isCompatible =
                  isStripQCompatible(itrack, layer_of_hits.is_barrel(), m_Err[iP], propPar, m_msErr, m_msPar);

              //rescale strip charge to track parameters and reapply the cut
              isCompatible &= passStripChargePCMfromTrack(
                  itrack, layer_of_hits.is_barrel(), charge_pcm[itrack], Hit::minChargePerCM(), propPar, m_msErr);
            }

            // Select only SiStrip hits with cluster size < maxClusterSize
            if (!layer_of_hits.is_pixel()) {
              if (layer_of_hits.refHit(m_XHitArr.At(itrack, hit_cnt, 0)).spanRows() >=
                  m_iteration_params->maxClusterSize) {
                isTooLargeCluster[itrack] = true;
                isCompatible = false;
              }
            }

            if (isCompatible) {
              CombCandidate &ccand = cloner.combCandWithOriginalIndex(m_SeedIdx(itrack, 0, 0));
              bool hitExists = false;
              int maxHits = m_NFoundHits(itrack, 0, 0);
              if (layer_of_hits.is_pixel()) {
                for (int i = 0; i <= maxHits; ++i) {
                  if (i > 2)
                    break;
                  if (ccand.hot(i).layer == layer_of_hits.layer_id()) {
                    hitExists = true;
                    break;
                  }
                }
              }
              if (hitExists)
                continue;

              nHitsAdded[itrack]++;
              const int hit_idx = m_XHitArr.At(itrack, hit_cnt, 0);

              // Register hit for overlap consideration, if chi2 cut is passed
              // To apply a fixed cut instead of dynamic cut for overlap: m_iteration_params->chi2CutOverlap
              if (chi2 < max_c2) {
                ccand[m_CandIdx(itrack, 0, 0)].considerHitForOverlap(
                    hit_idx, layer_of_hits.refHit(hit_idx).detIDinLayer(), chi2);
              }

              IdxChi2List tmpList;
              tmpList.trkIdx = m_CandIdx(itrack, 0, 0);
              tmpList.hitIdx = hit_idx;
              tmpList.module = layer_of_hits.refHit(hit_idx).detIDinLayer();
              tmpList.nhits = m_NFoundHits(itrack, 0, 0) + 1;
              tmpList.ntailholes = 0;
              tmpList.noverlaps = m_NOverlapHits(itrack, 0, 0);
              tmpList.nholes = num_all_minus_one_hits(itrack);
              tmpList.pt = std::abs(1.0f / m_Par[iP].At(itrack, 3, 0));
              tmpList.chi2 = m_Chi2(itrack, 0, 0) + chi2;
              tmpList.chi2_hit = chi2;
              tmpList.score = getScoreStruct(m_steering_params->m_track_scorer, tmpList);
              cloner.add_cand(m_SeedIdx(itrack, 0, 0) - offset, tmpList);

              dprint("  adding hit with hit_cnt=" << hit_cnt << " for trkIdx=" << tmpList.trkIdx
                                                  << " orig Seed=" << m_Label(itrack, 0, 0));
            }
          }
        }
      }

    }  //end loop over hits

    //now add invalid hit
    for (int itrack = 0; itrack < N_proc; ++itrack) {
      dprint("num_all_minus_one_hits(" << itrack << ")=" << num_all_minus_one_hits(itrack));

      // Cands that miss the layer are stashed away in MkBuilder(), before propagation,
      // and then merged back afterwards.
      if (m_XWsrResult[itrack].m_wsr == WSR_Outside) {
        continue;
      }

      // int fake_hit_idx = num_all_minus_one_hits(itrack) < m_iteration_params->maxHolesPerCand ? -1 : -2;
      int fake_hit_idx = ((num_all_minus_one_hits(itrack) < m_iteration_params->maxHolesPerCand) &&
                          (m_NTailMinusOneHits(itrack, 0, 0) < m_iteration_params->maxConsecHoles))
                             ? Hit::kHitMissIdx
                             : Hit::kHitStopIdx;

      if (m_XWsrResult[itrack].m_wsr == WSR_Edge) {
        fake_hit_idx = Hit::kHitEdgeIdx;
      }
      //now add fake hit for tracks that passsed through inactive modules
      else if (m_XWsrResult[itrack].m_in_gap == true && nHitsAdded[itrack] == 0) {
        fake_hit_idx = Hit::kHitInGapIdx;
      }
      //now add fake hit for cases where hit cluster size is larger than maxClusterSize
      else if (isTooLargeCluster[itrack] == true && nHitsAdded[itrack] == 0) {
        fake_hit_idx = Hit::kHitMaxClusterIdx;
      }

      // PROP-FAIL-ENABLE The following to be enabled when propagation failure
      // detection is properly implemented in propagate-to-R/Z.
      // // Override for failed propagation, this trumps all other cases.
      // if (m_XWsrResult[itrack].m_wsr == WSR_Failed) {
      //   fake_hit_idx = Hit::kHitStopIdx;
      // }

      IdxChi2List tmpList;
      tmpList.trkIdx = m_CandIdx(itrack, 0, 0);
      tmpList.hitIdx = fake_hit_idx;
      tmpList.module = -1;
      tmpList.nhits = m_NFoundHits(itrack, 0, 0);
      tmpList.ntailholes = (fake_hit_idx == Hit::kHitMissIdx ? m_NTailMinusOneHits(itrack, 0, 0) + 1
                                                             : m_NTailMinusOneHits(itrack, 0, 0));
      tmpList.noverlaps = m_NOverlapHits(itrack, 0, 0);
      tmpList.nholes = num_inside_minus_one_hits(itrack);
      tmpList.pt = std::abs(1.0f / m_Par[iP].At(itrack, 3, 0));
      tmpList.chi2 = m_Chi2(itrack, 0, 0);
      tmpList.chi2_hit = 0;
      tmpList.score = getScoreStruct(m_steering_params->m_track_scorer, tmpList);
      cloner.add_cand(m_SeedIdx(itrack, 0, 0) - offset, tmpList);
      dprint("adding invalid hit " << fake_hit_idx);
    }
  }

  //==============================================================================
  // UpdateWithLoadedHit
  //==============================================================================

  void MkFinder::updateWithLoadedHit(int N_proc, const LayerOfHits &layer_of_hits, const FindingFoos &fnd_foos) {
    // See comment in MkBuilder::find_tracks_in_layer() about intra / inter flags used here
    // for propagation to the hit.
    clearFailFlag();
    if (Config::usePropToPlane) {
      MPlexHV norm, dir;
      packModuleNormDir(layer_of_hits, 0, norm, dir, N_proc);
      kalmanPropagateAndUpdatePlane(m_Err[iP],
                                    m_Par[iP],
                                    m_Chg,
                                    m_msErr,
                                    m_msPar,
                                    norm,
                                    dir,
                                    m_Err[iC],
                                    m_Par[iC],
                                    m_FailFlag,
                                    N_proc,
                                    m_prop_config->finding_inter_layer_pflags,
                                    m_prop_config->finding_requires_propagation_to_hit_pos);
    } else {
      (*fnd_foos.m_update_param_foo)(m_Err[iP],
                                     m_Par[iP],
                                     m_Chg,
                                     m_msErr,
                                     m_msPar,
                                     m_Err[iC],
                                     m_Par[iC],
                                     m_FailFlag,
                                     N_proc,
                                     m_prop_config->finding_inter_layer_pflags,
                                     m_prop_config->finding_requires_propagation_to_hit_pos);
    }

    // PROP-FAIL-ENABLE The following to be enabled when propagation failure
    // detection is properly implemented in propagate-to-R/Z.
    // for (int i = 0; i < N_proc; ++i) {
    //   if (m_FailFlag[i]) {
    //     dprintf("MkFinder::updateWithLoadedHit fail in update, recovering.\n");
    //     m_Err[iC].copySlot(i, m_Err[iP]);
    //     m_Par[iC].copySlot(i, m_Par[iP]);
    //   }
    // }
  }

  void MkFinder::chi2OfLoadedHit(int N_proc, const FindingFoos &fnd_foos) {
    // We expect input in iC slots from above function.
    // See comment in MkBuilder::find_tracks_in_layer() about intra / inter flags used here
    // for propagation to the hit.
    clearFailFlag();
    (*fnd_foos.m_compute_chi2_foo)(m_Err[iC],
                                   m_Par[iC],
                                   m_Chg,
                                   m_msErr,
                                   m_msPar,
                                   m_Chi2,
                                   m_Par[iP],
                                   m_FailFlag,
                                   N_proc,
                                   m_prop_config->finding_inter_layer_pflags,
                                   m_prop_config->finding_requires_propagation_to_hit_pos);

    // PROP-FAIL-ENABLE .... removed here
  }

  //==============================================================================
  // CopyOutParErr
  //==============================================================================

  void MkFinder::copyOutParErr(std::vector<CombCandidate> &seed_cand_vec, int N_proc, bool outputProp) const {
    const int iO = outputProp ? iP : iC;

    for (int i = 0; i < N_proc; ++i) {
      TrackCand &cand = seed_cand_vec[m_SeedIdx(i, 0, 0)][m_CandIdx(i, 0, 0)];

      // Set the track state to the updated parameters
      m_Err[iO].copyOut(i, cand.errors_nc().Array());
      m_Par[iO].copyOut(i, cand.parameters_nc().Array());
      cand.setCharge(m_Chg(i, 0, 0));

      dprint((outputProp ? "propagated" : "updated")
             << " track parameters x=" << cand.parameters()[0] << " y=" << cand.parameters()[1]
             << " z=" << cand.parameters()[2] << " pt=" << 1. / cand.parameters()[3] << " posEta=" << cand.posEta());
    }
  }

  //==============================================================================
  // Backward Fit hack
  //==============================================================================

  void MkFinder::bkFitInputTracks(TrackVec &cands, int beg, int end) {
    // Uses HitOnTrack vector from Track directly + a local cursor array to current hit.

    MatriplexTrackPacker mtp(&cands[beg]);

    int itrack = 0;

    for (int i = beg; i < end; ++i, ++itrack) {
      const Track &trk = cands[i];

      m_Chg(itrack, 0, 0) = trk.charge();
      m_CurHit[itrack] = trk.nTotalHits() - 1;
      m_HoTArr[itrack] = trk.getHitsOnTrackArray();

      mtp.addInput(trk);
    }

    m_Chi2.setVal(0);

    mtp.pack(m_Err[iC], m_Par[iC]);

    m_Err[iC].scale(100.0f);
  }

  void MkFinder::bkFitInputTracks(EventOfCombCandidates &eocss, int beg, int end) {
    // Could as well use HotArrays from tracks directly + a local cursor array to last hit.

    // XXXX - shall we assume only TrackCand-zero is needed and that we can freely
    // bork the HoTNode array?

    MatriplexTrackPacker mtp(&eocss[beg][0]);

    int itrack = 0;

    for (int i = beg; i < end; ++i, ++itrack) {
      const TrackCand &trk = eocss[i][0];

      m_Chg(itrack, 0, 0) = trk.charge();
      m_CurNode[itrack] = trk.lastCcIndex();
      m_HoTNodeArr[itrack] = trk.combCandidate()->hotsData();

      // XXXX Need TrackCand* to update num-hits. Unless I collect info elsewhere
      // and fix it in BkFitOutputTracks.
      m_TrkCand[itrack] = &eocss[i][0];

      mtp.addInput(trk);
    }

    m_Chi2.setVal(0);

    mtp.pack(m_Err[iC], m_Par[iC]);

    m_Err[iC].scale(100.0f);
  }

  //------------------------------------------------------------------------------

  void MkFinder::bkFitOutputTracks(TrackVec &cands, int beg, int end, bool outputProp) {
    // Only copy out track params / errors / chi2, all the rest is ok.

    const int iO = outputProp ? iP : iC;

    int itrack = 0;
    for (int i = beg; i < end; ++i, ++itrack) {
      Track &trk = cands[i];

      m_Err[iO].copyOut(itrack, trk.errors_nc().Array());
      m_Par[iO].copyOut(itrack, trk.parameters_nc().Array());

      trk.setChi2(m_Chi2(itrack, 0, 0));
      if (isFinite(trk.chi2())) {
        trk.setScore(getScoreCand(m_steering_params->m_track_scorer, trk));
      }
    }
  }

  void MkFinder::bkFitOutputTracks(EventOfCombCandidates &eocss, int beg, int end, bool outputProp) {
    // Only copy out track params / errors / chi2, all the rest is ok.

    // XXXX - where will rejected hits get removed?

    const int iO = outputProp ? iP : iC;

    int itrack = 0;
    for (int i = beg; i < end; ++i, ++itrack) {
      TrackCand &trk = eocss[i][0];

      m_Err[iO].copyOut(itrack, trk.errors_nc().Array());
      m_Par[iO].copyOut(itrack, trk.parameters_nc().Array());

      trk.setChi2(m_Chi2(itrack, 0, 0));
      if (isFinite(trk.chi2())) {
        trk.setScore(getScoreCand(m_steering_params->m_track_scorer, trk));
      }
    }
  }

  //------------------------------------------------------------------------------

#if defined(DEBUG_BACKWARD_FIT) || defined(DEBUG_BACKWARD_FIT_BH)
  namespace {
    float e2s(float x) { return 1e4 * std::sqrt(x); }
  }  // namespace
#endif

  void MkFinder::bkFitFitTracksBH(const EventOfHits &eventofhits,
                                  const SteeringParams &st_par,
                                  const int N_proc,
                                  bool chiDebug) {
    // Prototyping final backward fit.
    // This works with track-finding indices, before remapping.
    //
    // Layers should be collected during track finding and list all layers that have actual hits.
    // Then we could avoid checking which layers actually do have hits.

    MPlexQF tmp_chi2;
    float tmp_err[6] = {666, 0, 666, 0, 0, 666};
    float tmp_pos[3];

    for (auto lp_iter = st_par.m_layer_plan.rbegin(); lp_iter != st_par.m_layer_plan.rend(); ++lp_iter) {
      const int layer = lp_iter->m_layer;

      const LayerOfHits &L = eventofhits[layer];
      const LayerInfo &LI = *L.layer_info();

      int count = 0;
      for (int i = 0; i < N_proc; ++i) {
        while (m_CurHit[i] >= 0 && m_HoTArr[i][m_CurHit[i]].index < 0)
          --m_CurHit[i];

        if (m_CurHit[i] >= 0 && m_HoTArr[i][m_CurHit[i]].layer == layer) {
          // Skip the overlap hits -- if they exist.
          // 1. Overlap hit gets placed *after* the original hit in TrackCand::exportTrack()
          // which is *before* in the reverse iteration that we are doing here.
          // 2. Seed-hit merging can result in more than two hits per layer.
          while (m_CurHit[i] > 0 && m_HoTArr[i][m_CurHit[i] - 1].layer == layer)
            --m_CurHit[i];

          const Hit &hit = L.refHit(m_HoTArr[i][m_CurHit[i]].index);
          m_msErr.copyIn(i, hit.errArray());
          m_msPar.copyIn(i, hit.posArray());
          ++count;
          --m_CurHit[i];
        } else {
          tmp_pos[0] = m_Par[iC](i, 0, 0);
          tmp_pos[1] = m_Par[iC](i, 1, 0);
          tmp_pos[2] = m_Par[iC](i, 2, 0);
          m_msErr.copyIn(i, tmp_err);
          m_msPar.copyIn(i, tmp_pos);
        }
      }

      if (count == 0)
        continue;

      // ZZZ Could add missing hits here, only if there are any actual matches.

      if (LI.is_barrel()) {
        propagateTracksToHitR(m_msPar, N_proc, m_prop_config->backward_fit_pflags);

        kalmanOperation(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                        m_Err[iP],
                        m_Par[iP],
                        m_msErr,
                        m_msPar,
                        m_Err[iC],
                        m_Par[iC],
                        tmp_chi2,
                        N_proc);
      } else {
        propagateTracksToHitZ(m_msPar, N_proc, m_prop_config->backward_fit_pflags);

        kalmanOperationEndcap(KFO_Calculate_Chi2 | KFO_Update_Params,
                              m_Err[iP],
                              m_Par[iP],
                              m_msErr,
                              m_msPar,
                              m_Err[iC],
                              m_Par[iC],
                              tmp_chi2,
                              N_proc);
      }

      //fixup invpt sign and charge
      for (int n = 0; n < N_proc; ++n) {
        if (m_Par[iC].At(n, 3, 0) < 0) {
          m_Chg.At(n, 0, 0) = -m_Chg.At(n, 0, 0);
          m_Par[iC].At(n, 3, 0) = -m_Par[iC].At(n, 3, 0);
        }
      }

#ifdef DEBUG_BACKWARD_FIT_BH
      // Dump per hit chi2
      for (int i = 0; i < N_proc; ++i) {
        float r_h = std::hypot(m_msPar.At(i, 0, 0), m_msPar.At(i, 1, 0));
        float r_t = std::hypot(m_Par[iC].At(i, 0, 0), m_Par[iC].At(i, 1, 0));

        // if ((std::isnan(tmp_chi2[i]) || std::isnan(r_t)))
        // if ( ! std::isnan(tmp_chi2[i]) && tmp_chi2[i] > 0) // && tmp_chi2[i] > 30)
        if (chiDebug) {
          int ti = iP;
          printf(
              "CHIHIT %3d %10g %10g %10g %10g %10g %11.5g %11.5g %11.5g %10g %10g %10g %10g %11.5g %11.5g %11.5g %10g "
              "%10g %10g %10g %10g %11.5g %11.5g\n",
              layer,
              tmp_chi2[i],
              m_msPar.At(i, 0, 0),
              m_msPar.At(i, 1, 0),
              m_msPar.At(i, 2, 0),
              r_h,  // x_h y_h z_h r_h -- hit pos
              e2s(m_msErr.At(i, 0, 0)),
              e2s(m_msErr.At(i, 1, 1)),
              e2s(m_msErr.At(i, 2, 2)),  // ex_h ey_h ez_h -- hit errors
              m_Par[ti].At(i, 0, 0),
              m_Par[ti].At(i, 1, 0),
              m_Par[ti].At(i, 2, 0),
              r_t,  // x_t y_t z_t r_t -- track pos
              e2s(m_Err[ti].At(i, 0, 0)),
              e2s(m_Err[ti].At(i, 1, 1)),
              e2s(m_Err[ti].At(i, 2, 2)),  // ex_t ey_t ez_t -- track errors
              1.0f / m_Par[ti].At(i, 3, 0),
              m_Par[ti].At(i, 4, 0),
              m_Par[ti].At(i, 5, 0),                                     // pt, phi, theta
              std::atan2(m_msPar.At(i, 1, 0), m_msPar.At(i, 0, 0)),      // phi_h
              std::atan2(m_Par[ti].At(i, 1, 0), m_Par[ti].At(i, 0, 0)),  // phi_t
              1e4f * std::hypot(m_msPar.At(i, 0, 0) - m_Par[ti].At(i, 0, 0),
                                m_msPar.At(i, 1, 0) - m_Par[ti].At(i, 1, 0)),  // d_xy
              1e4f * (m_msPar.At(i, 2, 0) - m_Par[ti].At(i, 2, 0))             // d_z
              // e2s((m_msErr.At(i,0,0) + m_msErr.At(i,1,1)) / (r_h * r_h)),     // ephi_h
              // e2s((m_Err[ti].At(i,0,0) + m_Err[ti].At(i,1,1)) / (r_t * r_t))  // ephi_t
          );
        }
      }
#endif

      // update chi2
      m_Chi2.add(tmp_chi2);
    }
  }

  //------------------------------------------------------------------------------

  void MkFinder::print_par_err(int corp, int mslot) const {
#ifdef DEBUG
    printf("Parameters:\n");
    for (int i = 0; i < 6; ++i) {
      printf("  %12.4g", m_Par[corp].constAt(mslot, i, 0));
    }
    printf("\nError matrix\n");
    for (int i = 0; i < 6; ++i) {
      for (int j = 0; j < 6; ++j) {
        printf("  %12.4g", m_Err[corp].constAt(mslot, i, j));
      }
      printf("\n");
    }
#endif
  }

  void MkFinder::bkFitFitTracks(const EventOfHits &eventofhits,
                                const SteeringParams &st_par,
                                const int N_proc,
                                bool chiDebug) {
    // Prototyping final backward fit.
    // This works with track-finding indices, before remapping.
    //
    // Layers should be collected during track finding and list all layers that have actual hits.
    // Then we could avoid checking which layers actually do have hits.

    // bool debug = true;

    MPlexQF tmp_chi2;
    MPlexQI no_mat_effs;
    float tmp_err[6] = {666, 0, 666, 0, 0, 666};
    float tmp_pos[3];

#if defined(DEBUG_PROP_UPDATE)
    const int DSLOT = 0;
    printf("bkfit entry, track in slot %d\n", DSLOT);
    print_par_err(iC, DSLOT);
#endif

    for (auto lp_iter = st_par.make_iterator(SteeringParams::IT_BkwFit); lp_iter.is_valid(); ++lp_iter) {
      const int layer = lp_iter.layer();

      const LayerOfHits &L = eventofhits[layer];
      const LayerInfo &LI = *L.layer_info();

#if defined(DEBUG_BACKWARD_FIT)
      const Hit *last_hit_ptr[NN];
#endif

      no_mat_effs.setVal(0);
      int done_count = 0;
      int here_count = 0;
      for (int i = 0; i < N_proc; ++i) {
        while (m_CurNode[i] >= 0 && m_HoTNodeArr[i][m_CurNode[i]].m_hot.index < 0) {
          m_CurNode[i] = m_HoTNodeArr[i][m_CurNode[i]].m_prev_idx;
        }

        if (m_CurNode[i] < 0)
          ++done_count;

        if (m_CurNode[i] >= 0 && m_HoTNodeArr[i][m_CurNode[i]].m_hot.layer == layer) {
          // Skip the overlap hits -- if they exist.
          // 1. Overlap hit gets placed *after* the original hit in TrackCand::exportTrack()
          // which is *before* in the reverse iteration that we are doing here.
          // 2. Seed-hit merging can result in more than two hits per layer.
          // while (m_CurHit[i] > 0 && m_HoTArr[ i ][ m_CurHit[i] - 1 ].layer == layer) --m_CurHit[i];
          while (m_HoTNodeArr[i][m_CurNode[i]].m_prev_idx >= 0 &&
                 m_HoTNodeArr[i][m_HoTNodeArr[i][m_CurNode[i]].m_prev_idx].m_hot.layer == layer)
            m_CurNode[i] = m_HoTNodeArr[i][m_CurNode[i]].m_prev_idx;

          const Hit &hit = L.refHit(m_HoTNodeArr[i][m_CurNode[i]].m_hot.index);

#ifdef DEBUG_BACKWARD_FIT
          last_hit_ptr[i] = &hit;
#endif
          m_msErr.copyIn(i, hit.errArray());
          m_msPar.copyIn(i, hit.posArray());
          ++here_count;

          m_CurNode[i] = m_HoTNodeArr[i][m_CurNode[i]].m_prev_idx;
        } else {
#ifdef DEBUG_BACKWARD_FIT
          last_hit_ptr[i] = nullptr;
#endif
          no_mat_effs[i] = 1;
          tmp_pos[0] = m_Par[iC](i, 0, 0);
          tmp_pos[1] = m_Par[iC](i, 1, 0);
          tmp_pos[2] = m_Par[iC](i, 2, 0);
          m_msErr.copyIn(i, tmp_err);
          m_msPar.copyIn(i, tmp_pos);
        }
      }

      if (done_count == N_proc)
        break;
      if (here_count == 0)
        continue;

      // ZZZ Could add missing hits here, only if there are any actual matches.

      clearFailFlag();

      // PROP-FAIL-ENABLE Once always "copy input to output on fail" is removed from
      // propagateToR one might want to enable this for barrel or endcap or both.
      if (LI.is_barrel()) {
        propagateTracksToHitR(m_msPar, N_proc, m_prop_config->backward_fit_pflags, &no_mat_effs);

        kalmanOperation(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                        m_Err[iP],
                        m_Par[iP],
                        m_msErr,
                        m_msPar,
                        m_Err[iC],
                        m_Par[iC],
                        tmp_chi2,
                        N_proc);
      } else {
        propagateTracksToHitZ(m_msPar, N_proc, m_prop_config->backward_fit_pflags, &no_mat_effs);

        kalmanOperationEndcap(KFO_Calculate_Chi2 | KFO_Update_Params,
                              m_Err[iP],
                              m_Par[iP],
                              m_msErr,
                              m_msPar,
                              m_Err[iC],
                              m_Par[iC],
                              tmp_chi2,
                              N_proc);
      }

#if defined(DEBUG_PROP_UPDATE)
      printf("\nbkfit at layer %d, track in slot %d -- fail=%d, had hit=%d (%g, %g, %g)\n",
             LI.layer_id(),
             DSLOT,
             m_FailFlag[DSLOT],
             1 - no_mat_effs[DSLOT],
             m_msPar(DSLOT, 0, 0),
             m_msPar(DSLOT, 1, 0),
             m_msPar(DSLOT, 2, 0));
      printf("Propagated:\n");
      print_par_err(iP, DSLOT);
      printf("Updated:\n");
      print_par_err(iC, DSLOT);
#endif

      // Fixup for failed propagation or invpt sign and charge.
      for (int i = 0; i < N_proc; ++i) {
        // PROP-FAIL-ENABLE The following to be enabled when propagation failure
        // detection is properly implemented in propagate-to-R/Z.
        // 1. The following code was only expecting barrel state to be restored.
        //      auto barrel_pf(m_prop_config->backward_fit_pflags);
        //      barrel_pf.copy_input_state_on_fail = true;
        // 2. There is also check on chi2, commented out to keep physics changes minimal.
        /*
        if (m_FailFlag[i] && LI.is_barrel()) {
          // Barrel pflags are set to include PF_copy_input_state_on_fail.
          // Endcap errors are immaterial here (relevant for fwd search), with prop error codes
          // one could do other things.
          // Are there also fail conditions in KalmanUpdate?
#ifdef DEBUG
          if (debug && g_debug) {
            dprintf("MkFinder::bkFitFitTracks prop fail: chi2=%f, layer=%d, label=%d. Recovering.\n",
                    tmp_chi2[i], LI.layer_id(), m_Label[i]);
            print_par_err(iC, i);
          }
#endif
          m_Err[iC].copySlot(i, m_Err[iP]);
          m_Par[iC].copySlot(i, m_Par[iP]);
        } else if (tmp_chi2[i] > 200 || tmp_chi2[i] < 0) {
#ifdef DEBUG
          if (debug && g_debug) {
            dprintf("MkFinder::bkFitFitTracks chi2 fail: chi2=%f, layer=%d, label=%d. Recovering.\n",
                    tmp_chi2[i], LI.layer_id(), m_Label[i]);
            print_par_err(iC, i);
          }
#endif
          // Go back to propagated state (at the current hit, the previous one is lost).
          m_Err[iC].copySlot(i, m_Err[iP]);
          m_Par[iC].copySlot(i, m_Par[iP]);
        }
        */
        // Fixup invpt sign and charge.
        if (m_Par[iC].At(i, 3, 0) < 0) {
          m_Chg.At(i, 0, 0) = -m_Chg.At(i, 0, 0);
          m_Par[iC].At(i, 3, 0) = -m_Par[iC].At(i, 3, 0);
        }
      }

#if defined(DEBUG_BACKWARD_FIT)
      // clang-format off
      bool debug = true;
      const char beg_cur_sep = '/'; // set to ' ' root parsable printouts
      for (int i = 0; i < N_proc; ++i) {
        if (chiDebug && last_hit_ptr[i]) {
          TrackCand &bb = *m_TrkCand[i];
          int ti = iP;
          float chi = tmp_chi2.At(i, 0, 0);
          float chi_prnt = std::isfinite(chi) ? chi : -9;

#if defined(MKFIT_STANDALONE)
          const MCHitInfo &mchi = m_event->simHitsInfo_[last_hit_ptr[i]->mcHitID()];

          dprintf("BKF_OVERLAP %d %d %d %d %d %d %d "
                  "%f%c%f %f %f%c%f %f %f %f %d %d %d %d "
                  "%f %f %f %f %f\n",
              m_event->evtID(),
#else
          dprintf("BKF_OVERLAP %d %d %d %d %d %d "
                  "%f%c%f %f %f%c%f %f %f %f %d %d %d "
                  "%f %f %f %f %f\n",
#endif
              bb.label(), (int)bb.prodType(), bb.isFindable(),
              layer, L.is_stereo(), L.is_barrel(),
              bb.pT(), beg_cur_sep, 1.0f / m_Par[ti].At(i, 3, 0),
              bb.posEta(),
              bb.posPhi(), beg_cur_sep, std::atan2(m_Par[ti].At(i, 1, 0), m_Par[ti].At(i, 0, 0)),
              std::hypot(m_Par[ti].At(i, 0, 0), m_Par[ti].At(i, 1, 0)),
              m_Par[ti].At(i, 2, 0),
              chi_prnt,
              std::isnan(chi), std::isfinite(chi), chi > 0,
#if defined(MKFIT_STANDALONE)
              mchi.mcTrackID(),
#endif
              // The following three can get negative / prouce nans in e2s.
              // std::abs the args for FPE hunt.
              e2s(std::abs(m_Err[ti].At(i, 0, 0))),
              e2s(std::abs(m_Err[ti].At(i, 1, 1))),
              e2s(std::abs(m_Err[ti].At(i, 2, 2))),  // sx_t sy_t sz_t -- track errors
              1e4f * std::hypot(m_msPar.At(i, 0, 0) - m_Par[ti].At(i, 0, 0),
                                m_msPar.At(i, 1, 0) - m_Par[ti].At(i, 1, 0)),  // d_xy
              1e4f * (m_msPar.At(i, 2, 0) - m_Par[ti].At(i, 2, 0))             // d_z
          );
        }
      }
      // clang-format on
#endif

      // update chi2
      m_Chi2.add(tmp_chi2);
    }
  }

  //------------------------------------------------------------------------------

  void MkFinder::bkFitPropTracksToPCA(const int N_proc) {
    propagateTracksToPCAZ(N_proc, m_prop_config->pca_prop_pflags);
  }

}  // end namespace mkfit
