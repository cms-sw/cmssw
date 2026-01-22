#include "MkFitter.h"

#include "KalmanUtilsMPlex.h"
#include "MatriplexPackers.h"

//#define DEBUG
//#define DEBUG_FIT
//#define DEBUG_FITi
//#define DEBUG_FIT_BKW
#include "Debug.h"

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

  void MkFitter::bkReFitInputTracks(TrackVec &cands, std::vector<int> inds, int beg, int end) {
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

    int index;
    if (cands[inds[beg]].nFoundHits() % 2 == 0)
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
      if (m_Chi2(itrack, 0, 0) != m_Chi2(itrack, 0, 0))
        std::cout << "nan " << itrack << "nan " << i << std::endl;
#endif
      if (m_Chi2(itrack, 0, 0) != m_Chi2(itrack, 0, 0))
        continue;  //trick for the nan so the track is not dead

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

      int local_m = m_CurHit[i];
      while (local_m >= 0) {
#ifdef DEBUG_FITi
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
          int sign = L.is_barrel() > 0 ? 1 : -1;
          r2z.push_back(std::make_pair(sign * R2, z));
          indices.push_back(local_m);
        }
        local_m--;
      }
      //continue working with the track
      std::vector<int> sorted_indices;
      float z0 = r2z.back().second;
      bool barrel_prev = false;
      std::map<float, std::vector<int>> index_RorZ;

      for (int i = 0; i < (int)indices.size(); i++) {
        bool barrel = r2z[i].first > 0;
        float sorting;
#ifdef DEBUG_FIT
        std::cout << "R2 " << r2z[i].first << " z " << r2z[i].second << " z0 " << z0 << std::endl;
#endif
        if (barrel)
          sorting = -std::fabs(r2z[i].first);
        else
          sorting = -std::fabs(r2z[i].second - z0);

        if (i == 0) {
          index_RorZ[sorting].push_back(indices[i]);
        } else {
          if (barrel == barrel_prev)  //add to segment in barrel or endcap
          {
            index_RorZ[sorting].push_back(indices[i]);
          } else  //switch barrel to endcap
          {
            for (const auto &iRZ : index_RorZ)
              for (auto iiRZ : iRZ.second)
                sorted_indices.push_back(iiRZ);                      //push back indices sorted for segment
            index_RorZ.erase(index_RorZ.begin(), index_RorZ.end());  //empty the map
            index_RorZ[sorting].push_back(indices[i]);               //start new segment
          }
        }
        barrel_prev = barrel;
      }
      for (const auto &iRZ : index_RorZ)  //final segment
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
        for (int i = 0; i < (int)indices.size(); i++) {
          std::cout << "R2 " << r2z[i].first << " z " << r2z[i].second << " z0 " << z0 << std::endl;
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
#ifdef DEBUG_FIT
          std::cout << "mi.pos[0] " << mi.pos[0] << " mi.pos[1] " << mi.pos[1] << " mi.pos[2] " << mi.pos[2]
                    << std::endl;
          std::cout << "pnt[0] " << pnt(i, 0, 0) << " pnt[1] " << pnt(i, 1, 0) << " pnt[2] " << pnt(i, 2, 0)
                    << std::endl;
          std::cout << "at the track " << i << " / " << N_proc << " check the material " << std::endl;
#endif
          if (hitIndex[i] < (int)indices.size() &&
              m_HoTArr[i][index].layer == m_HoTArr[i][indices[hitIndex[i]]].layer) {
            no_mat_effs[i] = 1;
#ifdef DEBUG_FIT
            std::cout << "at the hit " << index << " Remove the material because it was applied in hit "
                      << indices[hitIndex[i]] << std::endl;
            std::cout << "layers are " << m_HoTArr[i][index].layer << " and  "
                      << m_HoTArr[i][indices[hitIndex[i]]].layer << std::endl;
#endif
          }
#ifdef DEBUG_FIT
          else {
            std::cout << "at the hit " << index << " Don't remove the material" << std::endl;
            if (hitIndex[i] < (int)indices.size())
              std::cout << "layers are " << m_HoTArr[i][index].layer << " and  "
                        << m_HoTArr[i][indices[hitIndex[i]]].layer << std::endl;
          }
#endif
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

      kalmanPropagateAndUpdateAndChi2Plane(m_Err[i1],
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
                                           *refit_flags,
                                           true,
                                           &no_mat_effs,
                                           &do_cpe,
                                           m_cpe_corr_func);

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

    for (int h = 0; h < nFoundHits; ++h)  //first loop over the group - need to use the mplex here
    {
#ifdef DEBUG_FIT_BKW
      std::cout << "MY HIT " << h << " nFoundHits " << nFoundHits << std::endl;
#endif
      no_mat_effs.setVal(0);
      do_cpe.setVal(-1);

      for (int i = 0; i < N_proc; ++i)  //loop over tracks in group
      {
        auto indices = indices_R2Z[i];
        std::reverse(indices.begin(), indices.end());
        int index = indices[hitIndex[i] - 1];
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
#ifdef DEBUG_FIT_BKW
          std::cout << "mi.pos[0] " << mi.pos[0] << " mi.pos[1] " << mi.pos[1] << " mi.pos[2] " << mi.pos[2]
                    << std::endl;
          std::cout << "pnt[0] " << pnt(i, 0, 0) << " pnt[1] " << pnt(i, 1, 0) << " pnt[2] " << pnt(i, 2, 0)
                    << std::endl;
          std::cout << "at the track " << i << " / " << N_proc << " check the material " << std::endl;
#endif
          if (hitIndex[i] < (int)indices.size() &&
              m_HoTArr[i][index].layer == m_HoTArr[i][indices[hitIndex[i]]].layer) {
            no_mat_effs[i] = 1;
#ifdef DEBUG_FIT_BKW
            std::cout << "at the hit " << index << " Remove the material because it was applied in hit "
                      << indices[hitIndex[i]] << std::endl;
            std::cout << "layers are " << m_HoTArr[i][index].layer << " and  "
                      << m_HoTArr[i][indices[hitIndex[i]]].layer << std::endl;
#endif
          }
#ifdef DEBUG_FIT_BKW
          else {
            std::cout << "at the hit " << index << " Don't remove the material" << std::endl;
            if (hitIndex[i] < (int)indices.size())
              std::cout << "layers are " << m_HoTArr[i][index].layer << " and  "
                        << m_HoTArr[i][indices[hitIndex[i]]].layer << std::endl;
          }
#endif
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

      kalmanPropagateAndUpdateAndChi2Plane(m_Err[i1],
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
                                           *refit_flags,
                                           true,
                                           &no_mat_effs,
                                           &do_cpe,
                                           m_cpe_corr_func);

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

  //------------------------------------------------------------------------------

  void MkFitter::release() {
    m_event = nullptr;
    //refit_flags
    refit_flags = nullptr;
    //cpe
    m_cpe_corr_func = nullptr;
  }

}  // end namespace mkfit
