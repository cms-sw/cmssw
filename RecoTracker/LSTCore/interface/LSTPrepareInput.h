#ifndef RecoTracker_LSTCore_interface_LSTPrepareInput_h
#define RecoTracker_LSTCore_interface_LSTPrepareInput_h

#include <memory>
#include <Math/Vector3D.h>
#include <Math/VectorUtil.h>

#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"

namespace lst {

  inline ROOT::Math::XYZVector calculateR3FromPCA(const ROOT::Math::XYZVector& p3, float dxy, float dz) {
    const float pt = p3.rho();
    const float p = p3.r();
    const float vz = dz * pt * pt / p / p;

    const float vx = -dxy * p3.y() / pt - p3.x() / p * p3.z() / p * dz;
    const float vy = dxy * p3.x() / pt - p3.y() / p * p3.z() / p * dz;
    return {vx, vy, vz};
  }

  inline LSTInputHostCollection prepareInput(std::vector<float> const& see_px,
                                             std::vector<float> const& see_py,
                                             std::vector<float> const& see_pz,
                                             std::vector<float> const& see_dxy,
                                             std::vector<float> const& see_dz,
                                             std::vector<float> const& see_ptErr,
                                             std::vector<float> const& see_etaErr,
                                             std::vector<float> const& see_stateTrajGlbX,
                                             std::vector<float> const& see_stateTrajGlbY,
                                             std::vector<float> const& see_stateTrajGlbZ,
                                             std::vector<float> const& see_stateTrajGlbPx,
                                             std::vector<float> const& see_stateTrajGlbPy,
                                             std::vector<float> const& see_stateTrajGlbPz,
                                             std::vector<int> const& see_q,
                                             std::vector<std::vector<int>> const& see_hitIdx,
                                             std::vector<unsigned int> const& see_algo,
                                             std::vector<unsigned int> const& ph2_detId,
                                             std::vector<float> const& ph2_x,
                                             std::vector<float> const& ph2_y,
                                             std::vector<float> const& ph2_z,
#ifndef LST_STANDALONE
                                             std::vector<TrackingRecHit const*> const& ph2_hits,
#endif
                                             float const ptCut) {
    std::vector<float> trkX;
    std::vector<float> trkY;
    std::vector<float> trkZ;
    std::vector<unsigned int> hitId;
    std::vector<unsigned int> hitIdxs;
    std::vector<Params_pLS::ArrayUxHits> hitIndices_vec;
    std::vector<float> deltaPhi_vec;
    std::vector<float> ptIn_vec;
    std::vector<float> ptErr_vec;
    std::vector<float> px_vec;
    std::vector<float> py_vec;
    std::vector<float> pz_vec;
    std::vector<float> eta_vec;
    std::vector<float> etaErr_vec;
    std::vector<float> phi_vec;
    std::vector<int> charge_vec;
    std::vector<unsigned int> seedIdx_vec;
    std::vector<int> superbin_vec;
    std::vector<PixelType> pixelType_vec;
    std::vector<char> isQuad_vec;

    const int hit_size = ph2_x.size();

    unsigned int count = 0;
    auto n_see = see_stateTrajGlbPx.size();
    px_vec.reserve(n_see);
    py_vec.reserve(n_see);
    pz_vec.reserve(n_see);
    hitIndices_vec.reserve(n_see);
    ptIn_vec.reserve(n_see);
    ptErr_vec.reserve(n_see);
    etaErr_vec.reserve(n_see);
    eta_vec.reserve(n_see);
    phi_vec.reserve(n_see);
    charge_vec.reserve(n_see);
    seedIdx_vec.reserve(n_see);
    deltaPhi_vec.reserve(n_see);
    trkX.reserve(4 * n_see);
    trkY.reserve(4 * n_see);
    trkZ.reserve(4 * n_see);
    hitId.reserve(4 * n_see);
    hitIdxs.reserve(hit_size + 4 * n_see);
    hitIdxs.resize(hit_size);

    std::iota(hitIdxs.begin(), hitIdxs.end(), 0);

    for (size_t iSeed = 0; iSeed < n_see; iSeed++) {
      // Only needed for standalone
      bool good_seed_type = see_algo.empty() || see_algo[iSeed] == 4 || see_algo[iSeed] == 22;
      if (!good_seed_type)
        continue;

      ROOT::Math::XYZVector p3LH(see_stateTrajGlbPx[iSeed], see_stateTrajGlbPy[iSeed], see_stateTrajGlbPz[iSeed]);
      float ptIn = p3LH.rho();
      float eta = p3LH.eta();
      float ptErr = see_ptErr[iSeed];

      if ((ptIn > ptCut - 2 * ptErr)) {
        ROOT::Math::XYZVector r3LH(see_stateTrajGlbX[iSeed], see_stateTrajGlbY[iSeed], see_stateTrajGlbZ[iSeed]);
        ROOT::Math::XYZVector p3PCA(see_px[iSeed], see_py[iSeed], see_pz[iSeed]);
        ROOT::Math::XYZVector r3PCA(calculateR3FromPCA(p3PCA, see_dxy[iSeed], see_dz[iSeed]));

        // The charge could be used directly in the line below
        float pixelSegmentDeltaPhiChange = ROOT::Math::VectorUtil::DeltaPhi(p3LH, r3LH);
        float etaErr = see_etaErr[iSeed];
        float px = p3LH.x();
        float py = p3LH.y();
        float pz = p3LH.z();

        int charge = see_q[iSeed];
        PixelType pixtype = PixelType::kInvalid;

        if (ptIn >= 2.0)
          pixtype = PixelType::kHighPt;
        else if (ptIn >= (ptCut - 2 * ptErr) and ptIn < 2.0) {
          if (pixelSegmentDeltaPhiChange >= 0)
            pixtype = PixelType::kLowPtPosCurv;
          else
            pixtype = PixelType::kLowPtNegCurv;
        } else
          continue;

        unsigned int hitIdx0 = hit_size + count;
        count++;
        unsigned int hitIdx1 = hit_size + count;
        count++;
        unsigned int hitIdx2 = hit_size + count;
        count++;
        unsigned int hitIdx3;
        if (see_hitIdx[iSeed].size() <= 3)
          hitIdx3 = hitIdx2;
        else {
          hitIdx3 = hit_size + count;
          count++;
        }

        trkX.push_back(r3PCA.x());
        trkY.push_back(r3PCA.y());
        trkZ.push_back(r3PCA.z());
        trkX.push_back(p3PCA.rho());
        float p3PCA_Eta = p3PCA.eta();
        trkY.push_back(p3PCA_Eta);
        float p3PCA_Phi = p3PCA.phi();
        trkZ.push_back(p3PCA_Phi);
        trkX.push_back(r3LH.x());
        trkY.push_back(r3LH.y());
        trkZ.push_back(r3LH.z());
        hitId.push_back(1);
        hitId.push_back(1);
        hitId.push_back(1);
        if (see_hitIdx[iSeed].size() > 3) {
          trkX.push_back(r3LH.x());
          trkY.push_back(see_dxy[iSeed]);
          trkZ.push_back(see_dz[iSeed]);
          hitId.push_back(1);
        }
        px_vec.push_back(px);
        py_vec.push_back(py);
        pz_vec.push_back(pz);

        hitIndices_vec.push_back({{hitIdx0, hitIdx1, hitIdx2, hitIdx3}});
        ptIn_vec.push_back(ptIn);
        ptErr_vec.push_back(ptErr);
        etaErr_vec.push_back(etaErr);
        eta_vec.push_back(eta);
        float phi = p3LH.phi();
        phi_vec.push_back(phi);
        charge_vec.push_back(charge);
        seedIdx_vec.push_back(iSeed);
        deltaPhi_vec.push_back(pixelSegmentDeltaPhiChange);

        hitIdxs.push_back(see_hitIdx[iSeed][0]);
        hitIdxs.push_back(see_hitIdx[iSeed][1]);
        hitIdxs.push_back(see_hitIdx[iSeed][2]);
        char isQuad = false;
        if (see_hitIdx[iSeed].size() > 3) {
          isQuad = true;
          hitIdxs.push_back(see_hitIdx[iSeed][3]);
        }
        float neta = 25.;
        float nphi = 72.;
        float nz = 25.;
        int etabin = (p3PCA_Eta + 2.6) / ((2 * 2.6) / neta);
        int phibin = (p3PCA_Phi + std::numbers::pi_v<float>) / ((2. * std::numbers::pi_v<float>) / nphi);
        int dzbin = (std::clamp(see_dz[iSeed], -30.f, 30.f) + 30) / (2 * 30 / nz);
        int isuperbin = (nz * nphi) * etabin + (nz)*phibin + dzbin;
        superbin_vec.push_back(isuperbin);
        pixelType_vec.push_back(pixtype);
        isQuad_vec.push_back(isQuad);
      }
    }

    // Build the SoAs
    int nHitsOT = ph2_x.size();
    int nHitsIT = trkX.size();
    int nPixelHits = hitIndices_vec.size();
    int nPixelSeeds = ptIn_vec.size();
    if (static_cast<unsigned int>(nPixelSeeds) > n_max_pixel_segments_per_module) {
      nPixelSeeds = n_max_pixel_segments_per_module;
    }

    std::array<int, 3> const soa_sizes{{nHitsIT + nHitsOT, nPixelHits, nPixelSeeds}};
    LSTInputHostCollection lstInputHC(soa_sizes, cms::alpakatools::host());

    auto hits = lstInputHC.view<InputHitsSoA>();
    std::memcpy(hits.xs(), ph2_x.data(), nHitsOT * sizeof(float));
    std::memcpy(hits.ys(), ph2_y.data(), nHitsOT * sizeof(float));
    std::memcpy(hits.zs(), ph2_z.data(), nHitsOT * sizeof(float));
    std::memcpy(hits.detid(), ph2_detId.data(), nHitsOT * sizeof(unsigned int));
#ifndef LST_STANDALONE
    std::memcpy(hits.hits(), ph2_hits.data(), nHitsOT * sizeof(TrackingRecHit const*));
#endif

    std::memcpy(hits.xs() + nHitsOT, trkX.data(), nHitsIT * sizeof(float));
    std::memcpy(hits.ys() + nHitsOT, trkY.data(), nHitsIT * sizeof(float));
    std::memcpy(hits.zs() + nHitsOT, trkZ.data(), nHitsIT * sizeof(float));
    std::memcpy(hits.detid() + nHitsOT, hitId.data(), nHitsIT * sizeof(unsigned int));
#ifndef LST_STANDALONE
    std::memset(hits.hits() + nHitsOT, 0, nHitsIT * sizeof(TrackingRecHit const*));
#endif

    std::memcpy(hits.idxs(), hitIdxs.data(), (nHitsIT + nHitsOT) * sizeof(unsigned int));

    auto pixelHits = lstInputHC.view<InputPixelHitsSoA>();
    std::memcpy(pixelHits.hitIndices(), hitIndices_vec.data(), nPixelHits * sizeof(Params_pLS::ArrayUxHits));
    std::memcpy(pixelHits.deltaPhi(), deltaPhi_vec.data(), nPixelHits * sizeof(float));

    auto pixelSeeds = lstInputHC.view<InputPixelSeedsSoA>();
    std::memcpy(pixelSeeds.ptIn(), ptIn_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.ptErr(), ptErr_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.px(), px_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.py(), py_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.pz(), pz_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.etaErr(), etaErr_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.isQuad(), isQuad_vec.data(), nPixelSeeds * sizeof(char));
    std::memcpy(pixelSeeds.eta(), eta_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.phi(), phi_vec.data(), nPixelSeeds * sizeof(float));
    std::memcpy(pixelSeeds.charge(), charge_vec.data(), nPixelSeeds * sizeof(int));
    std::memcpy(pixelSeeds.seedIdx(), seedIdx_vec.data(), nPixelSeeds * sizeof(unsigned int));
    std::memcpy(pixelSeeds.superbin(), superbin_vec.data(), nPixelSeeds * sizeof(int));
    std::memcpy(pixelSeeds.pixelType(), pixelType_vec.data(), nPixelSeeds * sizeof(PixelType));

    return lstInputHC;
  }

}  // namespace lst

#endif
