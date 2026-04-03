/** Computation of input features for superclustering DNN. Used by plugins/TracksterLinkingBySuperClustering.cc and plugins/SuperclusteringSampleDumper.cc */
// Author: Theo Cuisset - theo.cuisset@cern.ch
// Date: 11/2023

// Modified by Gamze Sokmen - gamze.sokmen@cern.ch
// Changes: Implementation of the delta time feature under a new DNN input version (v3) for the superclustering DNN and correcting the seed pT calculation.
// Date: 07/2025

// Modified by Felice Pantaleo <felice.pantaleo@cern.ch>
// Improved memory usage and inference performance.
// Date: 02/2026

#include "RecoHGCal/TICL/interface/SuperclusteringDNNInputs.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>

#include <Math/Rotation3D.h>
#include <Math/Vector2D.h>
#include <Math/Vector3D.h>
#include <Math/VectorUtil.h>

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace ticl {

  namespace {
    template <class V1, class V2>
    inline float cosTheta2D(const V1& v1, const V2& v2) {
      const float v1_r2 = v1.X() * v1.X() + v1.Y() * v1.Y();
      const float v2_r2 = v2.X() * v2.X() + v2.Y() * v2.Y();
      const float denom = v1_r2 * v2_r2;
      if (denom <= 0.f) {
        return 0.f;
      }
      const float pdot = v1.X() * v2.X() + v1.Y() * v2.Y();
      const float c = pdot / std::sqrt(denom);
      return std::clamp(c, -1.f, 1.f);
    }

    template <class V1, class V2>
    inline float angle2D(const V1& v1, const V2& v2) {
      return std::acos(cosTheta2D(v1, v2));
    }

    inline float explainedVarianceRatio(const Trackster& ts) {
      const float denom =
          std::accumulate(std::begin(ts.eigenvalues()), std::end(ts.eigenvalues()), 0.f, std::plus<float>());
      if (denom == 0.f) {
        edm::LogWarning("HGCalTICLSuperclustering")
            << "Sum of eigenvalues was zero for trackster. Could not compute explained variance ratio.";
        return 0.f;
      }
      return ts.eigenvalues()[0] / denom;
    }

  }  // namespace

  void SuperclusteringDNNInputV1::computeInto(Trackster const& ts_base,
                                              Trackster const& ts_toCluster,
                                              std::span<float> out) const {
    assert(out.size() >= kNFeatures);

    const auto& bc_c = ts_toCluster.barycenter();
    const auto& bc_s = ts_base.barycenter();

    out[0] = std::abs(bc_c.Eta()) - std::abs(bc_s.Eta());
    out[1] = bc_c.Phi() - bc_s.phi();
    out[2] = ts_toCluster.raw_energy();
    out[3] = bc_c.Eta();
    out[4] = ts_toCluster.raw_pt();
    out[5] = bc_s.Eta();
    out[6] = bc_s.Phi();
    out[7] = ts_base.raw_energy();
    out[8] = ts_base.raw_pt();
  }

  void SuperclusteringDNNInputV2::computeInto(Trackster const& ts_base,
                                              Trackster const& ts_toCluster,
                                              std::span<float> out) const {
    assert(out.size() >= kNFeatures);

    using ROOT::Math::XYVectorF;
    using ROOT::Math::XYZVectorF;
    using ROOT::Math::VectorUtil::Angle;

    const auto& bc_c = ts_toCluster.barycenter();
    const auto& bc_s = ts_base.barycenter();

    const XYZVectorF& pca_seed_cmsFrame(ts_base.eigenvectors(0));
    const XYZVectorF& pca_cand_cmsFrame(ts_toCluster.eigenvectors(0));
    const XYZVectorF xs(pca_seed_cmsFrame.Cross(XYZVectorF(0, 0, 1)).Unit());
    const ROOT::Math::Rotation3D rot(xs, xs.Cross(pca_seed_cmsFrame).Unit(), pca_seed_cmsFrame);
    const XYZVectorF pca_cand_seedFrame = rot(pca_cand_cmsFrame);

    const float explVarRatio = explainedVarianceRatio(ts_toCluster);

    out[0] = std::abs(bc_c.Eta()) - std::abs(bc_s.Eta());
    out[1] = bc_c.Phi() - bc_s.phi();
    out[2] = ts_toCluster.raw_energy();
    out[3] = bc_c.Eta();
    out[4] = ts_toCluster.raw_pt();
    out[5] = bc_s.Eta();
    out[6] = bc_s.Phi();
    out[7] = ts_base.raw_energy();
    out[8] = ts_base.raw_pt();

    out[9] = Angle(pca_cand_cmsFrame, pca_seed_cmsFrame);

    out[10] = angle2D(XYVectorF(pca_cand_seedFrame.x(), pca_cand_seedFrame.z()), XYVectorF(0, 1));
    out[11] = angle2D(XYVectorF(pca_cand_seedFrame.y(), pca_cand_seedFrame.z()), XYVectorF(0, 1));

    out[12] = angle2D(XYVectorF(pca_cand_cmsFrame.x(), pca_cand_cmsFrame.y()),
                      XYVectorF(pca_seed_cmsFrame.x(), pca_seed_cmsFrame.y()));
    out[13] = angle2D(XYVectorF(pca_cand_cmsFrame.y(), pca_cand_cmsFrame.z()),
                      XYVectorF(pca_seed_cmsFrame.y(), pca_seed_cmsFrame.z()));
    out[14] = angle2D(XYVectorF(pca_cand_cmsFrame.x(), pca_cand_cmsFrame.z()),
                      XYVectorF(pca_seed_cmsFrame.x(), pca_seed_cmsFrame.z()));

    out[15] = ts_toCluster.eigenvalues()[0];
    out[16] = explVarRatio;
  }

  void SuperclusteringDNNInputV3::computeInto(Trackster const& ts_base,
                                              Trackster const& ts_toCluster,
                                              std::span<float> out) const {
    assert(out.size() >= kNFeatures);

    using ROOT::Math::XYVectorF;
    using ROOT::Math::XYZVectorF;
    using ROOT::Math::VectorUtil::Angle;

    const auto& bc_c = ts_toCluster.barycenter();
    const auto& bc_s = ts_base.barycenter();

    const XYZVectorF& pca_seed_cmsFrame(ts_base.eigenvectors(0));
    const XYZVectorF& pca_cand_cmsFrame(ts_toCluster.eigenvectors(0));
    const XYZVectorF xs(pca_seed_cmsFrame.Cross(XYZVectorF(0, 0, 1)).Unit());
    const ROOT::Math::Rotation3D rot(xs, xs.Cross(pca_seed_cmsFrame).Unit(), pca_seed_cmsFrame);
    const XYZVectorF pca_cand_seedFrame = rot(pca_cand_cmsFrame);

    const float explVarRatio = explainedVarianceRatio(ts_toCluster);

    const float raw_dt = ts_toCluster.time() - ts_base.time();
    const float mod_deltaTime = (raw_dt < -kDeltaTimeDefault || raw_dt > kDeltaTimeDefault) ? kBadDeltaTime : raw_dt;

    out[0] = std::abs(bc_c.Eta()) - std::abs(bc_s.Eta());
    out[1] = bc_c.Phi() - bc_s.phi();
    out[2] = ts_toCluster.raw_energy();
    out[3] = bc_c.Eta();
    out[4] = ts_toCluster.raw_pt();
    out[5] = bc_s.Eta();
    out[6] = bc_s.Phi();
    out[7] = ts_base.raw_energy();
    out[8] = ts_base.raw_pt();

    out[9] = Angle(pca_cand_cmsFrame, pca_seed_cmsFrame);

    out[10] = angle2D(XYVectorF(pca_cand_seedFrame.x(), pca_cand_seedFrame.z()), XYVectorF(0, 1));
    out[11] = angle2D(XYVectorF(pca_cand_seedFrame.y(), pca_cand_seedFrame.z()), XYVectorF(0, 1));

    out[12] = angle2D(XYVectorF(pca_cand_cmsFrame.x(), pca_cand_cmsFrame.y()),
                      XYVectorF(pca_seed_cmsFrame.x(), pca_seed_cmsFrame.y()));
    out[13] = angle2D(XYVectorF(pca_cand_cmsFrame.y(), pca_cand_cmsFrame.z()),
                      XYVectorF(pca_seed_cmsFrame.y(), pca_seed_cmsFrame.z()));
    out[14] = angle2D(XYVectorF(pca_cand_cmsFrame.x(), pca_cand_cmsFrame.z()),
                      XYVectorF(pca_seed_cmsFrame.x(), pca_seed_cmsFrame.z()));

    out[15] = ts_toCluster.eigenvalues()[0];
    out[16] = explVarRatio;
    out[17] = mod_deltaTime;
  }

  std::unique_ptr<AbstractSuperclusteringDNNInput> makeSuperclusteringDNNInputFromString(std::string dnnInputVersion) {
    if (dnnInputVersion == "v1") {
      return std::make_unique<SuperclusteringDNNInputV1>();
    }
    if (dnnInputVersion == "v2") {
      return std::make_unique<SuperclusteringDNNInputV2>();
    }
    if (dnnInputVersion == "v3") {
      return std::make_unique<SuperclusteringDNNInputV3>();
    }
    assert(false);
    return nullptr;
  }

}  // namespace ticl
