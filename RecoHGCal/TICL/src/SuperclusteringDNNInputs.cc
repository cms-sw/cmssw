/** Computation of input features for superclustering DNN. Used by plugins/TracksterLinkingBySuperClustering.cc and plugins/SuperclusteringSampleDumper.cc */
// Author: Theo Cuisset - theo.cuisset@cern.ch
// Date: 11/2023
#include "RecoHGCal/TICL/interface/SuperclusteringDNNInputs.h"

#include <algorithm>
#include <numeric>
#include <cmath>

#include <Math/Vector2D.h>
#include <Math/Vector3D.h>
#include <Math/Rotation3D.h>
#include <Math/VectorUtil.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

namespace ticl {

  std::vector<float> SuperclusteringDNNInputV1::computeVector(Trackster const& ts_base, Trackster const& ts_toCluster) {
    /*  We use the barycenter for most of the variables below as that is what seems to have been used by Alessandro Tarabini, 
        but using PCA might be better. 
        (It would need retraining of the DNN)
    */
    return {{
        std::abs(ts_toCluster.barycenter().Eta()) - std::abs(ts_base.barycenter().Eta()),  //DeltaEtaBaryc
        ts_toCluster.barycenter().Phi() - ts_base.barycenter().phi(),                      //DeltaPhiBaryc
        ts_toCluster.raw_energy(),                                                         //multi_en
        ts_toCluster.barycenter().Eta(),                                                   //multi_eta
        (ts_toCluster.raw_energy() * std::sin(ts_toCluster.barycenter().Theta())),         //multi_pt
        ts_base.barycenter().Eta(),                                                        //seedEta
        ts_base.barycenter().Phi(),                                                        //seedPhi
        ts_base.raw_energy(),                                                              //seedEn
        (ts_base.raw_energy() * std::sin(ts_toCluster.barycenter().Theta())),              //seedPt
    }};
  }

  // Helper functions for angles. Adapted from ROOT (3D vectors -> 2D vectors)
  template <class Vector1, class Vector2>
  float CosTheta2D(const Vector1& v1, const Vector2& v2) {
    float arg;
    float v1_r2 = v1.X() * v1.X() + v1.Y() * v1.Y();
    float v2_r2 = v2.X() * v2.X() + v2.Y() * v2.Y();
    float ptot2 = v1_r2 * v2_r2;
    if (ptot2 <= 0) {
      arg = 0.0;
    } else {
      float pdot = v1.X() * v2.X() + v1.Y() * v2.Y();
      using std::sqrt;
      arg = pdot / sqrt(ptot2);
      if (arg > 1.0)
        arg = 1.0;
      if (arg < -1.0)
        arg = -1.0;
    }
    return arg;
  }
  template <class Vector1, class Vector2>
  inline float Angle2D(const Vector1& v1, const Vector2& v2) {
    return static_cast<float>(std::acos(CosTheta2D(v1, v2)));
  }

  std::vector<float> SuperclusteringDNNInputV2::computeVector(Trackster const& ts_base, Trackster const& ts_toCluster) {
    using ROOT::Math::XYVectorF;
    using ROOT::Math::XYZVectorF;
    using ROOT::Math::VectorUtil::Angle;
    XYZVectorF const& pca_seed_cmsFrame(ts_base.eigenvectors(0));
    XYZVectorF const& pca_cand_cmsFrame(ts_toCluster.eigenvectors(0));
    XYZVectorF xs(pca_seed_cmsFrame.Cross(XYZVectorF(0, 0, 1)).Unit());
    ROOT::Math::Rotation3D rot(xs, xs.Cross(pca_seed_cmsFrame).Unit(), pca_seed_cmsFrame);

    XYZVectorF pca_cand_seedFrame = rot(pca_cand_cmsFrame);  // seed coordinates

    float explVar_denominator = std::accumulate(
        std::begin(ts_toCluster.eigenvalues()), std::end(ts_toCluster.eigenvalues()), 0.f, std::plus<float>());
    float explVarRatio = 0.;
    if (explVar_denominator != 0.) {
      explVarRatio = ts_toCluster.eigenvalues()[0] / explVar_denominator;
    } else {
      edm::LogWarning("HGCalTICLSuperclustering")
          << "Sum of eigenvalues was zero for trackster. Could not compute explained variance ratio.";
    }

    return {{
        std::abs(ts_toCluster.barycenter().Eta()) - std::abs(ts_base.barycenter().Eta()),  //DeltaEtaBaryc
        ts_toCluster.barycenter().Phi() - ts_base.barycenter().phi(),                      //DeltaPhiBaryc
        ts_toCluster.raw_energy(),                                                         //multi_en
        ts_toCluster.barycenter().Eta(),                                                   //multi_eta
        (ts_toCluster.raw_energy() * std::sin(ts_toCluster.barycenter().Theta())),         //multi_pt
        ts_base.barycenter().Eta(),                                                        //seedEta
        ts_base.barycenter().Phi(),                                                        //seedPhi
        ts_base.raw_energy(),                                                              //seedEn
        (ts_base.raw_energy() * std::sin(ts_toCluster.barycenter().Theta())),              //seedPt
        static_cast<float>(Angle(pca_cand_cmsFrame, pca_seed_cmsFrame)),  // theta : angle between seed and candidate
        Angle2D(XYVectorF(pca_cand_seedFrame.x(), pca_cand_seedFrame.z()), XYVectorF(0, 1)),  // theta_xz_seedFrame
        Angle2D(XYVectorF(pca_cand_seedFrame.y(), pca_cand_seedFrame.z()), XYVectorF(0, 1)),  // theta_yz_seedFrame
        Angle2D(XYVectorF(pca_cand_cmsFrame.x(), pca_cand_cmsFrame.y()),
                XYVectorF(pca_seed_cmsFrame.x(), pca_seed_cmsFrame.y())),  // theta_xy_cmsFrame
        Angle2D(XYVectorF(pca_cand_cmsFrame.y(), pca_cand_cmsFrame.z()),
                XYVectorF(pca_seed_cmsFrame.y(), pca_seed_cmsFrame.z())),  // theta_yz_cmsFrame
        Angle2D(XYVectorF(pca_cand_cmsFrame.x(), pca_cand_cmsFrame.z()),
                XYVectorF(pca_seed_cmsFrame.x(), pca_seed_cmsFrame.z())),  // theta_xz_cmsFrame
        ts_toCluster.eigenvalues()[0],                                     // explVar
        explVarRatio                                                       // explVarRatio
    }};
  }

  std::unique_ptr<AbstractSuperclusteringDNNInput> makeSuperclusteringDNNInputFromString(std::string dnnInputVersion) {
    if (dnnInputVersion == "v1")
      return std::make_unique<SuperclusteringDNNInputV1>();
    else if (dnnInputVersion == "v2")
      return std::make_unique<SuperclusteringDNNInputV2>();
    assert(false);
  }

}  // namespace ticl