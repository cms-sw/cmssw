#ifndef RecoTracker_LSTCore_interface_LSTInput_h
#define RecoTracker_LSTCore_interface_LSTInput_h

#include <memory>
#include "Math/Vector3D.h"
#include "Math/VectorUtil.h"

#include "RecoTracker/LSTCore/interface/Common.h"

namespace lst {

  ROOT::Math::XYZVector calculateR3FromPCA(const ROOT::Math::XYZVector& p3, float dxy, float dz) {
    const float pt = p3.rho();
    const float p = p3.r();
    const float vz = dz * pt * pt / p / p;

    const float vx = -dxy * p3.y() / pt - p3.x() / p * p3.z() / p * dz;
    const float vy = dxy * p3.x() / pt - p3.y() / p * p3.z() / p * dz;
    return {vx, vy, vz};
  }

  std::tuple<std::unique_ptr<HitsHostCollection>, std::unique_ptr<PixelSegmentsHostCollection>> prepareInput(
      std::vector<float> const& see_px,
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
      std::vector<unsigned int> const& ph2_detId,
      std::vector<float> const& ph2_x,
      std::vector<float> const& ph2_y,
      std::vector<float> const& ph2_z,
      float const ptCut) {
    std::vector<float> trkX;
    std::vector<float> trkY;
    std::vector<float> trkZ;
    std::vector<unsigned int> hitId;
    std::vector<unsigned int> hitIdxs;
    std::vector<unsigned int> hitIndices_vec0;
    std::vector<unsigned int> hitIndices_vec1;
    std::vector<unsigned int> hitIndices_vec2;
    std::vector<unsigned int> hitIndices_vec3;
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

    unsigned int count = 0;
    auto n_see = see_stateTrajGlbPx.size();
    px_vec.reserve(n_see);
    py_vec.reserve(n_see);
    pz_vec.reserve(n_see);
    hitIndices_vec0.reserve(n_see);
    hitIndices_vec1.reserve(n_see);
    hitIndices_vec2.reserve(n_see);
    hitIndices_vec3.reserve(n_see);
    ptIn_vec.reserve(n_see);
    ptErr_vec.reserve(n_see);
    etaErr_vec.reserve(n_see);
    eta_vec.reserve(n_see);
    phi_vec.reserve(n_see);
    charge_vec.reserve(n_see);
    seedIdx_vec.reserve(n_see);
    deltaPhi_vec.reserve(n_see);
    trkX = ph2_x;
    trkY = ph2_y;
    trkZ = ph2_z;
    hitId = ph2_detId;
    hitIdxs.resize(ph2_detId.size());

    std::iota(hitIdxs.begin(), hitIdxs.end(), 0);
    const int hit_size = trkX.size();

    for (size_t iSeed = 0; iSeed < n_see; iSeed++) {
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

        hitIndices_vec0.push_back(hitIdx0);
        hitIndices_vec1.push_back(hitIdx1);
        hitIndices_vec2.push_back(hitIdx2);
        hitIndices_vec3.push_back(hitIdx3);
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
        int dzbin = (see_dz[iSeed] + 30) / (2 * 30 / nz);
        int isuperbin = (nz * nphi) * etabin + (nz)*phibin + dzbin;
        superbin_vec.push_back(isuperbin);
        pixelType_vec.push_back(pixtype);
        isQuad_vec.push_back(isQuad);
      }
    }

    // Build the SoAs
    int nHits = trkX.size();
    int nPixelHits = hitIndices_vec0.size();
    std::array<int, 2> const hits_sizes{{nHits, nPixelHits}};
    auto hitsHC = std::make_unique<HitsHostCollection>(hits_sizes, cms::alpakatools::host());

    auto hits = hitsHC->view<HitsSoA>();
    std::memcpy(hits.xs(), trkX.data(), nHits * sizeof(float));
    std::memcpy(hits.ys(), trkY.data(), nHits * sizeof(float));
    std::memcpy(hits.zs(), trkZ.data(), nHits * sizeof(float));
    std::memcpy(hits.detid(), hitId.data(), nHits * sizeof(unsigned int));
    std::memcpy(hits.idxs(), hitIdxs.data(), nHits * sizeof(unsigned int));

    auto pixelHits = hitsHC->view<PixelHitsSoA>();
    std::memcpy(pixelHits.hitIndices0(), hitIndices_vec0.data(), nPixelHits * sizeof(unsigned int));
    std::memcpy(pixelHits.hitIndices1(), hitIndices_vec1.data(), nPixelHits * sizeof(unsigned int));
    std::memcpy(pixelHits.hitIndices2(), hitIndices_vec2.data(), nPixelHits * sizeof(unsigned int));
    std::memcpy(pixelHits.hitIndices3(), hitIndices_vec3.data(), nPixelHits * sizeof(unsigned int));
    std::memcpy(pixelHits.deltaPhi(), deltaPhi_vec.data(), nPixelHits * sizeof(float));

    int pixelSegmentsSize = ptIn_vec.size();
    // if (pixelSegmentsSize > n_max_pixel_segments_per_module) {
    // lstWarning(
    //     "\
    //     *********************************************************\n\
    //     * Warning: Pixel line segments will be truncated.       *\n\
    //     * You need to increase n_max_pixel_segments_per_module. *\n\
    //     *********************************************************");
    //   pixelSegmentsSize = n_max_pixel_segments_per_module;
    // }

    // pixelModuleIndex_ = pixelMapping_.pixelModuleIndex;

    auto pixelSegmentsHC = std::make_unique<PixelSegmentsHostCollection>(pixelSegmentsSize, cms::alpakatools::host());
    PixelSegments pixelSegments = pixelSegmentsHC->view();
    std::memcpy(pixelSegments.ptIn(), ptIn_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.ptErr(), ptErr_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.px(), px_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.py(), py_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.pz(), pz_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.etaErr(), etaErr_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.isQuad(), isQuad_vec.data(), pixelSegmentsSize * sizeof(char));
    std::memcpy(pixelSegments.eta(), eta_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.phi(), phi_vec.data(), pixelSegmentsSize * sizeof(float));
    std::memcpy(pixelSegments.charge(), charge_vec.data(), pixelSegmentsSize * sizeof(int));
    std::memcpy(pixelSegments.seedIdx(), seedIdx_vec.data(), pixelSegmentsSize * sizeof(unsigned int));
    std::memcpy(pixelSegments.superbin(), superbin_vec.data(), pixelSegmentsSize * sizeof(int));
    std::memcpy(pixelSegments.pixelType(), pixelType_vec.data(), pixelSegmentsSize * sizeof(PixelType));

    return std::move(std::make_tuple(std::move(hitsHC), std::move(pixelSegmentsHC)));
  }

}  // namespace lst

#endif
