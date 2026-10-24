#include "MkFitter.h"

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "KalmanUtilsMPlex.h"
#include "FinalFit.h"
#include "MatriplexPackers.h"

//#define DEBUG
//#define DEBUG_FIT
//#define DEBUG_FIT_BKW
#include "Debug.h"

#include <cmath>
#include <sstream>

namespace mkfit {

  void MkFitter::fwdFitInputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end) {
    // Uses HitOnTrack vector from Track directly + a local cursor array to current hit.
#ifdef DEBUG_FIT_BKW
    std::cout << " -- fwdFitInputTracks " << std::endl;
#endif
    MatriplexTrackPacker mtp(&cands[inds[beg]]);

    int itrack = 0;

    for (int i = beg; i < end; ++i, ++itrack) {
      const Track &trk = cands[inds[i]];
      m_Chg(itrack, 0, 0) = trk.charge();
      m_CurHit[itrack] = trk.nTotalHits() - 1;  //I have to use in reverse... otherwise n hits unknown
      m_HoTArr[itrack] = trk.getHitsOnTrackArray();
#ifdef DEBUG_FIT
      std::cout << "trk pt " << trk.pT() << " trk eta " << trk.momEta() << std::endl;
      std::cout << "trk nTotalHits " << trk.nTotalHits() << " trk nFoundHits " << trk.nFoundHits() << std::endl;
#endif
      mtp.addInput(trk);
    }

    m_Chi2.setVal(0);
    mtp.pack(m_Err[iC], m_Par[iC]);
    m_Err[iC].scale(100.0f);
  }

  void MkFitter::bkReFitInputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end, int nFoundHits) {
    // Uses HitOnTrack vector from Track directly + a local cursor array to current hit.
#ifdef DEBUG_FIT_BKW
    std::cout << " -- bkReFitInputTracks " << std::endl;
#endif
    MatriplexTrackPacker mtp(&cands[inds[beg]]);

    int itrack = 0;

    for (int i = beg; i < end; ++i, ++itrack) {
      const Track &trk = cands[inds[i]];
      m_Chg(itrack, 0, 0) = trk.charge();
      m_CurHit[itrack] = trk.nTotalHits() - 1;
      m_HoTArr[itrack] = trk.getHitsOnTrackArray();
#ifdef DEBUG_FIT_BKW
      std::cout << "trk pt " << trk.pT() << " trk eta " << trk.momEta() << std::endl;
      std::cout << "trk nTotalHits " << trk.nTotalHits() << " trk nFoundHits " << trk.nFoundHits() << std::endl;
#endif
      mtp.addInput(trk);
    }

    m_Chi2.setVal(0);

    // the slot the forward pass ended in: it fitted the group's nFoundHits hits
    int index;
    if (nFoundHits % 2 == 0)
      index = iC;
    else
      index = iP;

    mtp.pack(m_Err[index], m_Par[index]);
    m_Err[index].scale(100.0f);
  }

  void MkFitter::reFitOutputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end, int nFoundHits, bool bkw) {
    // Only copy out track params / errors / chi2
    if (bkw)
      nFoundHits = nFoundHits * 2;
    int iO;
    if (nFoundHits % 2 == 0)
      iO = iC;
    else
      iO = iP;

    int itrack = 0;
    for (int i = beg; i < end; ++i, ++itrack) {
      Track &trk = cands[inds[i]];
#ifdef DEBUG_FIT
      if (bkw)
        std::cout << "before trk pt " << trk.pT() << " trk eta " << trk.momEta() << std::endl;
      if (bkw)
        std::cout << "before trk nTotalHits " << trk.nTotalHits() << " trk nFoundHits " << trk.nFoundHits()
                  << std::endl;
      if (!bkw)
        std::cout << "before trk pt " << trk.pT() << " trk eta " << trk.momEta() << std::endl;
      if (!bkw)
        std::cout << "before trk nTotalHits " << trk.nTotalHits() << " trk nFoundHits " << trk.nFoundHits()
                  << std::endl;
      if (!isFinite(m_Chi2(itrack, 0, 0)))
        std::cout << "nan " << itrack << "nan " << i << std::endl;
#endif
      // isFinite, not x != x: -Ofast assumes finite math and folds x != x to false.  A failed forward pass keeps the
      // track's previous state, from which the backward pass refits it; a failed backward pass, which gives the
      // track's final state, is a failed fit: the chi2 is set negative, and the converter drops the candidate
      // (whether or not per-hit states are stored)
      if (!isFinite(m_Chi2(itrack, 0, 0))) {
        if (bkw)
          trk.setChi2(-1.f);
        continue;
      }

      m_Err[iO].copyOut(itrack, trk.errors_nc().Array());
      m_Par[iO].copyOut(itrack, trk.parameters_nc().Array());

#ifdef DEBUG_FIT_BKW
      if (bkw)
        std::cout << "oout trk pt " << trk.pT() << " trk eta " << trk.momEta() << std::endl;
      if (bkw)
        std::cout << "oout trk nTotalHits " << trk.nTotalHits() << " trk nFoundHits " << trk.nFoundHits() << std::endl;
      if (bkw)
        std::cout << "mchi2 " << m_Chi2(itrack, 0, 0) << std::endl;
#endif

#ifdef DEBUG_FIT
      if (!bkw)
        std::cout << "oout trk pt " << trk.pT() << " trk eta " << trk.momEta() << std::endl;
      if (!bkw)
        std::cout << "oout trk nTotalHits " << trk.nTotalHits() << " trk nFoundHits " << trk.nFoundHits() << std::endl;
      if (!bkw)
        std::cout << "mchi2 " << m_Chi2(itrack, 0, 0) << std::endl;
#endif

      trk.setChi2(m_Chi2(itrack, 0, 0));
    }
  }

  //------------------------------------------------------------------------------

  std::vector<std::vector<int>> MkFitter::reFitIndices(const EventOfHits &eventofhits,
                                                       const int N_proc,
                                                       int nFoundHits) {
    std::vector<std::vector<int>> indices_R2Z;

    for (int i = 0; i < N_proc; ++i) {
      std::vector<std::pair<float, float>> r2z;
      std::vector<int> indices;

      float minR2 = std::numeric_limits<float>::max();
      float z_minR = 0;

      int local_m = m_CurHit[i];
      while (local_m >= 0) {
#ifdef DEBUG_FIT
        std::cout << " i layer " << m_HoTArr[i][local_m].layer << " i index " << m_HoTArr[i][local_m].index
                  << std::endl;
#endif
        if (m_HoTArr[i][local_m].index >= 0) {
          const LayerOfHits &L = eventofhits[m_HoTArr[i][local_m].layer];
          const Hit &hit = L.refHit(m_HoTArr[i][local_m].index);

          float x, y, z;
          x = hit.posArray()[0];
          y = hit.posArray()[1];
          z = hit.posArray()[2];
          float R2 = x * x + y * y;  // + z * z;
#ifdef DEBUG_FIT
          std::cout << "x " << x << " y " << y << " z " << z << std::endl;
          std::cout << L.is_barrel() << " layer of hits is barrel ... " << m_HoTArr[i][local_m].layer << std::endl;
          std::cout << R2 << " R2 -- z " << z << " index " << local_m << " i/Nproc " << i << " / " << N_proc
                    << std::endl;
#endif
          // NB: the sort below uses R2 and z only -- barrel/endcap does not enter it.
          r2z.push_back(std::make_pair(R2, z));
          indices.push_back(local_m);
          if (R2 < minR2) {
            minR2 = R2;
            z_minR = z;
          }
        }
        local_m--;
      }
      //continue working with the track
      std::map<float, std::vector<int>> index_RorZ;
      std::vector<int> sorted_indices;
      for (int i = 0; i < (int)r2z.size(); i++) {
        float r2 = r2z[i].first;
        float z = r2z[i].second - z_minR;
#ifdef DEBUG_FIT
        std::cout << "SORTING by 3dR" << "R2 " << r2z[i].first << " z " << r2z[i].second << " z0 " << z_minR
                  << " check " << -r2 - z * z << std::endl;
#endif
        index_RorZ[-r2 - z * z].push_back(indices[i]);
      }
      for (const auto &iRZ : index_RorZ)  //one segment
        for (auto iiRZ : iRZ.second)
          sorted_indices.push_back(iiRZ);

      if (indices.size() != sorted_indices.size()) {
        std::cout << indices.size() << "  " << sorted_indices.size() << std::endl;
        for (auto ii : indices) {
          std::cout << " indices ii " << ii << std::endl;
        }
        for (auto ii : sorted_indices) {
          std::cout << " indices ssii " << ii << std::endl;
        }
      }
#ifdef DEBUG_FIT
      std::cout << " check_size " << (indices.size() == sorted_indices.size()) << std::endl;
      for (auto ii : indices) {
        std::cout << " indices ii " << ii << std::endl;
      }
      for (auto ii : sorted_indices) {
        std::cout << " indices ssii " << ii << std::endl;
      }
#endif
      indices_R2Z.push_back(sorted_indices);  //sorted_indices
    }
    return indices_R2Z;
  }

  void MkFitter::fwdFitFitTracks(const EventOfHits &eventofhits,
                                 const int N_proc,
                                 int nFoundHits,
                                 std::vector<std::vector<int>> indices_R2Z,
                                 float *chi2) {
    MPlexQF outChi2(0.0f);
    MPlexLV propPar;

    MPlexHV norm, dir, pnt;
    MPlexQF mat_radl{0.0f}, mat_bbxi{0.0f};  // the module's own material (FinalFitFlags::material_per_module)
    const final_fit::ModuleMaterial modMat{mat_radl, mat_bbxi};

    MPlexQI no_mat_effs;
    MPlexQI do_cpe;

    no_mat_effs.setVal(0);
    do_cpe.setVal(-1);
#ifdef DEBUG_FIT
    const int DSLOT = 0;
    printf("fit entry, track in slot %d\n", DSLOT);
    printf("\ninitial fit , track in slot %d --- (%g, %g, %g)\n",
           DSLOT,
           m_msPar(DSLOT, 0, 0),
           m_msPar(DSLOT, 1, 0),
           m_msPar(DSLOT, 2, 0));
    printf(
        "\ninitial fit , track in slot %d --- (%g, %g, %g)\n", 1, m_msPar(1, 0, 0), m_msPar(1, 1, 0), m_msPar(1, 2, 0));
    printf(
        "\ninitial fit , track in slot %d --- (%g, %g, %g)\n", 2, m_msPar(2, 0, 0), m_msPar(2, 1, 0), m_msPar(2, 2, 0));
#endif

    int i1 = iC;  //local copy
    int i2 = iP;  //local copy

    if ((m_storeHitStates || m_storeOuter) && (int)m_fwdLocPar.size() < nFoundHits) {
      m_fwdLocPar.resize(nFoundHits);
      m_fwdLocErr.resize(nFoundHits);
      m_fwdPzSign.resize(nFoundHits);
    }

    int hitIndex[N_proc];

    for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
    {
      hitIndex[i] = indices_R2Z[i].size();
    }

    for (int h = 0; h < nFoundHits; ++h)  //first loop over the group - need to use the mplex here
    {
#ifdef DEBUG_FIT
      std::cout << "MY HIT " << h << " nFoundHits " << nFoundHits << std::endl;
#endif
      no_mat_effs.setVal(0);
      do_cpe.setVal(-1);

      for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
      {
        auto &indices = indices_R2Z[i];
        int index = indices[hitIndex[i] - 1];
#ifdef DEBUG_FIT
        std::cout << "DEBUG hitIndex " << hitIndex[i] << std::endl;
        std::cout << "DEBUG i " << index << std::endl;
        std::cout << "DEBUG i layer " << m_HoTArr[i][index].layer << std::endl;
        std::cout << "DEBUG i index " << m_HoTArr[i][index].index << std::endl;
#endif
        if (m_HoTArr[i][index].index >= 0) {  //should be a redundant check
          const LayerOfHits &L = eventofhits[m_HoTArr[i][index].layer];
          const Hit &hit = L.refHit(m_HoTArr[i][index].index);
          if (L.is_pixel()) {
            do_cpe[i] = m_HoTArr[i][index].index;
          }  //hopefully ok to get the cluster
#ifdef DEBUG_FIT
          std::cout << "m_msPar " << m_msPar(i, 0, 0) << std::endl;
#endif
          m_msErr.copyIn(i, hit.errArray());
          m_msPar.copyIn(i, hit.posArray());
#ifdef DEBUG_FIT
          std::cout << "hit.posArray()[0] " << hit.posArray()[0] << " hit.posArray()[1] " << hit.posArray()[1]
                    << " hit.posArray()[2] " << hit.posArray()[2] << std::endl;
          std::cout << "m_msPar " << m_msPar(i, 0, 0) << std::endl;
#endif
          unsigned int mid = hit.detIDinLayer();
          const ModuleInfo &mi = L.layer_info().module_info(mid);
          norm.At(i, 0, 0) = mi.zdir[0];
          norm.At(i, 1, 0) = mi.zdir[1];
          norm.At(i, 2, 0) = mi.zdir[2];
          dir.At(i, 0, 0) = mi.xdir[0];
          dir.At(i, 1, 0) = mi.xdir[1];
          dir.At(i, 2, 0) = mi.xdir[2];
          pnt.At(i, 0, 0) = mi.pos[0];
          pnt.At(i, 1, 0) = mi.pos[1];
          pnt.At(i, 2, 0) = mi.pos[2];
          mat_radl.At(i, 0, 0) = mi.radl;
          mat_bbxi.At(i, 0, 0) = mi.bbxi;
#ifdef DEBUG_FIT
          std::cout << "mi.pos[0] " << mi.pos[0] << " mi.pos[1] " << mi.pos[1] << " mi.pos[2] " << mi.pos[2]
                    << std::endl;
          std::cout << "pnt[0] " << pnt(i, 0, 0) << " pnt[1] " << pnt(i, 1, 0) << " pnt[2] " << pnt(i, 2, 0)
                    << std::endl;
          std::cout << "at the track " << i << " / " << N_proc << " check the material " << std::endl;
          std::cout << "at the hit " << index << " Don't remove the material" << std::endl;
          if (hitIndex[i] < (int)indices.size())
            std::cout << "layers are " << m_HoTArr[i][index].layer << " and  "
                      << m_HoTArr[i][indices[hitIndex[i]]].layer << std::endl;
#endif
          // the pass propagates to its first hit too, without material there: its start state need not lie on
          // that hit's plane (the built track, or the previous fit of a track refitted after outlier removal)
          if (index == indices.back())
            no_mat_effs[i] = 1;
        }
        hitIndex[i]--;
      }  //end of track by track loop

#ifdef DEBUG_FIT
      for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
      {
        std::cout << "right before propagation at hit " << h << " index NP " << i + 1 << "/" << N_proc << std::endl;
        std::cout << "update parameters" << std::endl;
        std::cout << "propagated track parameters x=" << m_Par[i1].constAt(i, 0, 0)
                  << " y=" << m_Par[i1].constAt(i, 1, 0) << " z=" << m_Par[i1].constAt(i, 2, 0) << std::endl;
        std::cout << "               hit position x=" << m_msPar.constAt(i, 0, 0) << " y=" << m_msPar.constAt(i, 1, 0)
                  << " z=" << m_msPar.constAt(i, 2, 0) << std::endl;
        std::cout << "   updated track parameters x=" << m_Par[i2].constAt(i, 0, 0)
                  << " y=" << m_Par[i2].constAt(i, 1, 0) << " z=" << m_Par[i2].constAt(i, 2, 0) << std::endl;
        std::cout << "tmp_chi2[i]" << outChi2[i] << std::endl;
        std::cout << norm.At(i, 0, 0) << " " << norm.At(i, 1, 0) << " " << norm.At(i, 2, 0) << " "
                  << "NORM" << std::endl;
        std::cout << dir.At(i, 0, 0) << " " << dir.At(i, 1, 0) << " " << dir.At(i, 2, 0) << " "
                  << "DIR" << std::endl;
        std::cout << pnt.At(i, 0, 0) << " " << pnt.At(i, 1, 0) << " " << pnt.At(i, 2, 0) << " "
                  << "PNT" << std::endl;
        std::cout << "index / Nproc " << i << " material flag " << no_mat_effs[i] << std::endl;
      }
#endif

      // per-hit states: keep the updated state in the module's local frame (at the innermost hit, h = 0, only as
      // the fallback where the backward one cannot be used); the outer state needs it at the outermost hit only
      final_fit::LocalStatesOut fwdLoc;
      const bool keepLoc = m_storeHitStates || (m_storeOuter && h == nFoundHits - 1);
      if (keepLoc) {
        fwdLoc.updPar = &m_fwdLocPar[h];
        fwdLoc.updErr = &m_fwdLocErr[h];
        fwdLoc.pzSign = &m_fwdPzSign[h];
      }
      final_fit::propagate_update(m_Err[i1],
                                  m_Par[i1],
                                  m_Chg,
                                  m_msErr,
                                  m_msPar,
                                  norm,
                                  dir,
                                  pnt,
                                  m_Err[i2],
                                  m_Par[i2],
                                  m_FailFlag,
                                  outChi2,
                                  N_proc,
                                  final_fit::Pass{*refit_flags, *refit_ffflags, true},
                                  &no_mat_effs,
                                  &do_cpe,
                                  m_cpe_corr_func,
                                  &modMat,
                                  keepLoc ? &fwdLoc : nullptr);
      if (keepLoc && h == nFoundHits - 1)
        m_fwdChi2Outer = outChi2;

#ifdef DEBUG_FIT
      std::cout << " i1 " << i1 << " iP " << iP << " iC " << iC << std::endl;
      std::cout << "++++++++++++++++++++++++++\n" << std::endl;
      for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
      {
        std::cout << "right after propagation at hit " << h << " index NP " << i + 1 << "/" << N_proc << std::endl;
        std::cout << "update parameters" << std::endl;
        std::cout << "propagated track parameters x=" << m_Par[i1].constAt(i, 0, 0)
                  << " y=" << m_Par[i1].constAt(i, 1, 0) << " z=" << m_Par[i1].constAt(i, 2, 0) << std::endl;
        std::cout << "               hit position x=" << m_msPar.constAt(i, 0, 0) << " y=" << m_msPar.constAt(i, 1, 0)
                  << " z=" << m_msPar.constAt(i, 2, 0) << std::endl;
        std::cout << "   updated track parameters x=" << m_Par[i2].constAt(i, 0, 0)
                  << " y=" << m_Par[i2].constAt(i, 1, 0) << " z=" << m_Par[i2].constAt(i, 2, 0) << std::endl;
        std::cout << "tmp_chi2[i]" << outChi2[i] << std::endl;
      }
#endif
      std::swap(i1, i2);

      // update chi2
      m_Chi2.add(outChi2);
      for (int i = 0; i < N_proc; ++i) {
        chi2[h + i * nFoundHits] = outChi2[i];
      }
    }  //end of loop over n hits
  }  //end of fit func

  void MkFitter::bkReFitFitTracks(const EventOfHits &eventofhits,
                                  const int N_proc,
                                  int nFoundHits,
                                  std::vector<std::vector<int>> indices_R2Z,
                                  float *chi2) {
#ifdef DEBUG_FIT_BKW
    std::cout << "bkReFitFitTracks " << nFoundHits << std::endl;
#endif
    MPlexQF outChi2;
    MPlexLV propPar;

    MPlexHV norm, dir, pnt;
    MPlexQF mat_radl{0.0f}, mat_bbxi{0.0f};  // the module's own material (FinalFitFlags::material_per_module)
    const final_fit::ModuleMaterial modMat{mat_radl, mat_bbxi};

    MPlexQI no_mat_effs;
    MPlexQI do_cpe;

    no_mat_effs.setVal(0);
    do_cpe.setVal(-1);

    int i1, i2;
    if (nFoundHits % 2 == 0) {
      i1 = iC;
      i2 = iP;
    } else {
      i1 = iP;
      i2 = iC;
    }

    int hitIndex[N_proc];

    for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
    {
      hitIndex[i] = indices_R2Z[i].size();
    }

    // FinalFitFlags::bkw_ms_fixed_momentum: the multiple-scattering noise of the whole backward pass at the |p| of
    // its start state (the forward result), fixed per track, not at the running estimate.
    float ms_ref_p[NN];
    if (refit_ffflags->bkw_ms_fixed_momentum) {
      for (int i = 0; i < NN; ++i) {
        const float ipt = m_Par[i1].constAt(i, 3, 0), sT = std::sin(m_Par[i1].constAt(i, 5, 0));
        ms_ref_p[i] = (i < N_proc && ipt > 0.f && sT > 0.f) ? 1.f / (ipt * sT) : 1.f;
      }
    }
    const final_fit::Pass pass{*refit_flags,
                               *refit_ffflags,
                               false,
                               refit_ffflags->bkw_ms_fixed_momentum ? ms_ref_p : nullptr,
                               refit_ffflags->bkw_sub_steps};

    for (int h = 0; h < nFoundHits; ++h)  //first loop over the group - need to use the mplex here
    {
#ifdef DEBUG_FIT_BKW
      std::cout << "MY HIT " << h << " nFoundHits " << nFoundHits << std::endl;
#endif
      no_mat_effs.setVal(0);
      do_cpe.setVal(-1);
      int bkHot[NN];  // HitOnTrack position of each lane's hit (per-hit states)

      for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
      {
        // The backward pass walks the same sorted list outer->inner.  Index it from
        // the back directly: reverse(v)[k-1] == v[v.size()-k].  Copying and reversing
        // the vector for every hit of every track produced the same indices.
        const auto &indices = indices_R2Z[i];
        const int nidx = (int)indices.size();
        int index = indices[nidx - hitIndex[i]];
        bkHot[i] = index;
#ifdef DEBUG_FIT_BKW
        std::cout << "DEBUG hitIndex " << hitIndex[i] << std::endl;
        std::cout << "DEBUG i " << index << std::endl;
        std::cout << "DEBUG i layer " << m_HoTArr[i][index].layer << std::endl;
        std::cout << "DEBUG i index " << m_HoTArr[i][index].index << std::endl;
#endif
        if (m_HoTArr[i][index].index >= 0) {  //should be a redundant check
          const LayerOfHits &L = eventofhits[m_HoTArr[i][index].layer];
          const Hit &hit = L.refHit(m_HoTArr[i][index].index);
          if (L.is_pixel()) {
            do_cpe[i] = m_HoTArr[i][index].index;
          }  //hopefully ok to get the cluster
#ifdef DEBUG_FIT_BKW
          std::cout << "m_msPar " << m_msPar(i, 0, 0) << std::endl;
#endif
          m_msErr.copyIn(i, hit.errArray());
          m_msPar.copyIn(i, hit.posArray());
#ifdef DEBUG_FIT_BKW
          std::cout << "hit.posArray()[0] " << hit.posArray()[0] << " hit.posArray()[1] " << hit.posArray()[1]
                    << " hit.posArray()[2] " << hit.posArray()[2] << std::endl;
          std::cout << "m_msPar " << m_msPar(i, 0, 0) << std::endl;
#endif
          unsigned int mid = hit.detIDinLayer();
          const ModuleInfo &mi = L.layer_info().module_info(mid);
          norm.At(i, 0, 0) = mi.zdir[0];
          norm.At(i, 1, 0) = mi.zdir[1];
          norm.At(i, 2, 0) = mi.zdir[2];
          dir.At(i, 0, 0) = mi.xdir[0];
          dir.At(i, 1, 0) = mi.xdir[1];
          dir.At(i, 2, 0) = mi.xdir[2];
          pnt.At(i, 0, 0) = mi.pos[0];
          pnt.At(i, 1, 0) = mi.pos[1];
          pnt.At(i, 2, 0) = mi.pos[2];
          mat_radl.At(i, 0, 0) = mi.radl;
          mat_bbxi.At(i, 0, 0) = mi.bbxi;
#ifdef DEBUG_FIT_BKW
          std::cout << "mi.pos[0] " << mi.pos[0] << " mi.pos[1] " << mi.pos[1] << " mi.pos[2] " << mi.pos[2]
                    << std::endl;
          std::cout << "pnt[0] " << pnt(i, 0, 0) << " pnt[1] " << pnt(i, 1, 0) << " pnt[2] " << pnt(i, 2, 0)
                    << std::endl;
          std::cout << "at the track " << i << " / " << N_proc << " check the material " << std::endl;

          std::cout << "at the hit " << index << " Don't remove the material" << std::endl;
          if (hitIndex[i] < nidx)
            std::cout << "layers are " << m_HoTArr[i][index].layer << " and  "
                      << m_HoTArr[i][indices[nidx - 1 - hitIndex[i]]].layer << std::endl;
#endif
          if (index == indices.front())  // the outermost hit
            no_mat_effs[i] = 1;
        }
        hitIndex[i]--;
      }  //end of track by track loop

#ifdef DEBUG_FIT_BKW
      for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
      {
        std::cout << "right before propagation at hit " << h << " index NP " << i + 1 << "/" << N_proc << std::endl;
        std::cout << "update parameters" << std::endl;
        std::cout << "propagated track parameters x=" << m_Par[i1].constAt(i, 0, 0)
                  << " y=" << m_Par[i1].constAt(i, 1, 0) << " z=" << m_Par[i1].constAt(i, 2, 0) << std::endl;
        std::cout << "               hit position x=" << m_msPar.constAt(i, 0, 0) << " y=" << m_msPar.constAt(i, 1, 0)
                  << " z=" << m_msPar.constAt(i, 2, 0) << std::endl;
        std::cout << "   updated track parameters x=" << m_Par[i2].constAt(i, 0, 0)
                  << " y=" << m_Par[i2].constAt(i, 1, 0) << " z=" << m_Par[i2].constAt(i, 2, 0) << std::endl;
        std::cout << "outChi2 " << outChi2[i] << std::endl;
        std::cout << norm.At(i, 0, 0) << " " << norm.At(i, 1, 0) << " " << norm.At(i, 2, 0) << " "
                  << "NORM" << std::endl;
        std::cout << dir.At(i, 0, 0) << " " << dir.At(i, 1, 0) << " " << dir.At(i, 2, 0) << " "
                  << "DIR" << std::endl;
        std::cout << pnt.At(i, 0, 0) << " " << pnt.At(i, 1, 0) << " " << pnt.At(i, 2, 0) << " "
                  << "PNT" << std::endl;
      }
#endif

      // per-hit states: the backward predicted state (between the hits) or the updated one (innermost hit, and the
      // outermost one where the forward pass failed), in the module's local frame as the update computes it
      const bool innermost = h == nFoundHits - 1;
      MPlex5V bkPredPar, bkUpdPar;
      MPlex5S bkPredErr, bkUpdErr;
      MPlexQI bkPzSign;
      final_fit::LocalStatesOut bkLoc;
      const bool keepBkLoc = m_storeHitStates || (m_storeOuter && h == 0);
      if (keepBkLoc) {
        if (m_storeHitStates && ((h > 0 && !innermost) || m_validateHitStates)) {
          bkLoc.predPar = &bkPredPar;
          bkLoc.predErr = &bkPredErr;
        }
        if ((m_storeHitStates && innermost) || h == 0) {
          bkLoc.updPar = &bkUpdPar;
          bkLoc.updErr = &bkUpdErr;
        }
        bkLoc.pzSign = &bkPzSign;
      }
      final_fit::propagate_update(m_Err[i1],
                                  m_Par[i1],
                                  m_Chg,
                                  m_msErr,
                                  m_msPar,
                                  norm,
                                  dir,
                                  pnt,
                                  m_Err[i2],
                                  m_Par[i2],
                                  m_FailFlag,
                                  outChi2,
                                  N_proc,
                                  pass,
                                  &no_mat_effs,
                                  &do_cpe,
                                  m_cpe_corr_func,
                                  &modMat,
                                  keepBkLoc ? &bkLoc : nullptr);

      if (keepBkLoc)
        storeHitStates(h, nFoundHits, N_proc, bkHot, bkPredPar, bkPredErr, bkUpdPar, bkUpdErr, bkPzSign, outChi2);

#ifdef DEBUG_FIT_BKW
      std::cout << " i1 " << i1 << " iP " << iP << " iC " << iC << std::endl;

      std::cout << "++++++++++++++++++++++++++\n" << std::endl;
      for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
      {
        std::cout << "right after propagation at hit " << h << " index NP " << i + 1 << "/" << N_proc << std::endl;
        std::cout << "update parameters" << std::endl;
        std::cout << "propagated track parameters x=" << m_Par[i1].constAt(i, 0, 0)
                  << " y=" << m_Par[i1].constAt(i, 1, 0) << " z=" << m_Par[i1].constAt(i, 2, 0) << std::endl;
        std::cout << "               hit position x=" << m_msPar.constAt(i, 0, 0) << " y=" << m_msPar.constAt(i, 1, 0)
                  << " z=" << m_msPar.constAt(i, 2, 0) << std::endl;
        std::cout << "   updated track parameters x=" << m_Par[i2].constAt(i, 0, 0)
                  << " y=" << m_Par[i2].constAt(i, 1, 0) << " z=" << m_Par[i2].constAt(i, 2, 0) << std::endl;
        std::cout << "outChi2 " << outChi2[i] << std::endl;
      }
#endif
      std::swap(i1, i2);

      // update chi2
      m_Chi2.add(outChi2);
      for (int i = 0; i < N_proc; ++i) {
        chi2[h + i * nFoundHits] = outChi2[i];
      }

    }  //end of loop over n hits
  }  //end of fit func

  void MkFitter::storeHitStates(const int h,
                                const int nFoundHits,
                                const int N_proc,
                                const int *hot,
                                const MPlex5V &bkPredPar,
                                const MPlex5S &bkPredErr,
                                const MPlex5V &bkUpdPar,
                                const MPlex5S &bkUpdErr,
                                const MPlexQI &bkPzSign,
                                const MPlexQF &bkChi2) {
    // The smoothed state at the h-th hit of the backward pass, as KFTrajectorySmoother: the forward updated
    // state at the outermost hit, the backward updated state at the innermost one, and in between the combination
    // of the forward updated and the backward predicted state.
    const int hf = nFoundHits - 1 - h;  // the same hit in the forward pass
    const bool outermost = h == 0, innermost = h == nFoundHits - 1;
    MPlex5V sPar;
    MPlex5S sErr;
    MPlexQI ok;
    const MPlex5V *xs = &sPar;
    const MPlex5S *cs = &sErr;
    const MPlexQI *pz = &bkPzSign;
    signed char kind = HitStateOnTrack::Combined;
    if (outermost) {
      xs = &m_fwdLocPar[hf];
      cs = &m_fwdLocErr[hf];
      pz = &m_fwdPzSign[hf];
      kind = HitStateOnTrack::ForwardOnly;
    } else if (innermost) {
      xs = &bkUpdPar;
      cs = &bkUpdErr;
      kind = HitStateOnTrack::BackwardOnly;
    } else {
      final_fit::smooth_local_states(m_fwdLocPar[hf], m_fwdLocErr[hf], bkPredPar, bkPredErr, sPar, sErr, ok, N_proc);
    }
    auto fill = [](HitStateOnTrack &o, const MPlex5V &par, const MPlex5S &err, int n) {
      bool fin = true;
      for (int i = 0; i < 5; ++i) {
        o.par[i] = par.constAt(n, i, 0);
        for (int j = 0; j <= i; ++j)
          o.err[i * (i + 1) / 2 + j] = err.constAt(n, i, j);
        const float d = o.err[i * (i + 3) / 2];
        fin = fin && isFinite(o.par[i]) && isFinite(d) && d > 0.f;
      }
      return fin;
    };
    for (int n = 0; n < N_proc; ++n) {
      // the per-hit state, or at the outermost hit without per-hit states only the outer state
      OuterHitState *outer = outermost ? m_outerOut[n] : nullptr;
      HitStateOnTrack *po = m_hsOut[n] ? &m_hsOut[n][hot[n]] : outer ? &outer->state : nullptr;
      if (po) {
        HitStateOnTrack &o = *po;
        o.valid = fill(o, *xs, *cs, n) && (kind != HitStateOnTrack::Combined || ok.constAt(n, 0, 0));
        o.pzSign = pz->constAt(n, 0, 0);
        o.kind = kind;
        // a smoothed state that cannot be used (the combination failed, a variance is not positive, or the forward
        // pass failed and the backward pass refitted from the track's previous state): the forward updated state at
        // this hit instead, else the backward one (updated at the outermost hit, predicted between the end hits), so
        // that the track keeps a usable state at every hit
        if (!o.valid && !outermost) {
          o.valid = fill(o, m_fwdLocPar[hf], m_fwdLocErr[hf], n);
          o.pzSign = m_fwdPzSign[hf].constAt(n, 0, 0);
          o.kind = HitStateOnTrack::ForwardOnly;
        }
        if (!o.valid && !innermost) {
          o.valid = outermost ? fill(o, bkUpdPar, bkUpdErr, n) : fill(o, bkPredPar, bkPredErr, n);
          o.pzSign = bkPzSign.constAt(n, 0, 0);
          o.kind = HitStateOnTrack::BackwardOnly;
        }
        o.chi2 = outermost ? m_fwdChi2Outer.constAt(n, 0, 0) : bkChi2.constAt(n, 0, 0);
        if (outer) {
          outer->state = o;
          outer->hot = hot[n];
        }
      }
      if (m_validateHitStates && m_hsFwdOut[n] && m_hsBwdOut[n]) {
        HitStateOnTrack &of = m_hsFwdOut[n][hot[n]];
        of.valid = fill(of, m_fwdLocPar[hf], m_fwdLocErr[hf], n);
        of.pzSign = m_fwdPzSign[hf].constAt(n, 0, 0);
        of.kind = kind;
        HitStateOnTrack &ob = m_hsBwdOut[n][hot[n]];
        ob.valid = fill(ob, bkPredPar, bkPredErr, n);
        ob.pzSign = bkPzSign.constAt(n, 0, 0);
        ob.kind = kind;
      }
    }
  }

  //------------------------------------------------------------------------------

  void MkFitter::release() {
    m_event = nullptr;
    //refit_flags
    refit_flags = nullptr;
    refit_ffflags = nullptr;
    //cpe
    m_cpe_corr_func = nullptr;
  }

}  // end namespace mkfit
