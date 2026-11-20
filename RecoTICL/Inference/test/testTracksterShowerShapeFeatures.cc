// The shower-shape features and the layer-cluster profile of hand-made tracksters have the documented order and values,
// the same values for both z signs, and the documented values without a main axis or without energy.

#include <array>
#include <cmath>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <Eigen/Core>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByShowerShape.h"
#include "RecoTICL/Inference/interface/TracksterLayerClusterProfile.h"

namespace {

  reco::CaloCluster layerCluster(float x, float y, float z, float energy) {
    return reco::CaloCluster(
        energy, math::XYZPoint(x, y, z), reco::CaloID(reco::CaloID::DET_HGCAL_ENDCAP), reco::CaloCluster::hgcal_em);
  }

  // A trackster of the layer clusters 0 ... nVertices - 1, with multiplicity 1 unless given.
  ticl::Trackster tracksterOf(unsigned nVertices,
                              float rawEnergy,
                              const ticl::Trackster::Vector& barycenter,
                              const std::vector<float>& multiplicities = {}) {
    ticl::Trackster trackster;
    for (unsigned i = 0; i < nVertices; ++i) {
      trackster.vertices().push_back(i);
      trackster.vertex_multiplicity().push_back(multiplicities.empty() ? 1.f : multiplicities[i]);
    }
    trackster.setRawEnergy(rawEnergy);
    trackster.setBarycenter(barycenter);
    return trackster;
  }

  // Sets the PCA: main axis `axis` in the yz plane, eigenvalues {4, 1, 0.25}, sigmas {1, 2, 3} and PCA sigmas
  // {0.5, 1, 1.5}.
  void setPCA(ticl::Trackster& trackster, const Eigen::Vector3f& axis) {
    Eigen::Matrix3f eigenvectors;
    eigenvectors.col(0) = axis;
    eigenvectors.col(1) = Eigen::Vector3f(1.f, 0.f, 0.f);
    eigenvectors.col(2) = Eigen::Vector3f(0.f, axis.z(), -axis.y());
    trackster.fillPCAVariables(Eigen::Vector3f(4.f, 1.f, 0.25f),
                               eigenvectors,
                               Eigen::Vector3f(1.f, 4.f, 9.f),
                               Eigen::Vector3f(0.25f, 1.f, 2.25f),
                               3,
                               ticl::Trackster::PCAOrdering::descending);
  }

  // Raw energy 10 GeV, CE-E energy 6 GeV, barycenter (30, 40, 320 zSign), 3 layer clusters and main axis
  // (0, 0.6, 0.8 zSign).
  ticl::Trackster showerTrackster(float zSign) {
    auto trackster = tracksterOf(3, 10.f, {30.f, 40.f, 320.f * zSign});
    trackster.setRawEmEnergy(6.f);
    setPCA(trackster, Eigen::Vector3f(0.f, 0.6f, 0.8f * zSign));
    return trackster;
  }

  template <std::size_t N>
  void checkEqual(const std::array<float, N>& values, const std::array<float, N>& expected) {
    for (std::size_t k = 0; k < N; ++k) {
      INFO("index " << k);
      CHECK(values[k] == Catch::Approx(expected[k]).margin(1e-5));
    }
  }

  const float kLogMinEigenvalue = std::log(1e-6f);

}  // namespace

TEST_CASE("Shower-shape features of a trackster") {
  using Features = std::array<float, ticl::TracksterInferenceByShowerShape::kNFeatures>;

  SECTION("Order and values of the features") {
    const float etaBarycenter = std::asinh(320.f / 50.f);
    const float etaAxis = std::asinh(0.8f / 0.6f);
    const Features expected{std::log1p(10.f),
                            0.6f,
                            etaBarycenter,
                            320.f,
                            std::log(4.f),
                            std::log(4.f),
                            0.f,
                            std::log(0.25f),
                            4.f / 5.25f,
                            1.f,
                            2.f,
                            3.f,
                            0.5f,
                            1.f,
                            1.5f,
                            etaBarycenter - etaAxis};
    checkEqual(ticl::TracksterInferenceByShowerShape::features(showerTrackster(1.f)), expected);
  }

  SECTION("Both z signs give the same features") {
    checkEqual(ticl::TracksterInferenceByShowerShape::features(showerTrackster(-1.f)),
               ticl::TracksterInferenceByShowerShape::features(showerTrackster(1.f)));
  }

  SECTION("A zero main axis gives |eta| of the barycenter as the last feature") {
    for (float zSign : {1.f, -1.f}) {
      const auto trackster = tracksterOf(1, 5.f, {30.f, 40.f, 320.f * zSign});
      const Features expected{std::log1p(5.f),
                              0.f,
                              std::asinh(320.f / 50.f),
                              320.f,
                              std::log(2.f),
                              kLogMinEigenvalue,
                              kLogMinEigenvalue,
                              kLogMinEigenvalue,
                              1.f / 3.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              0.f,
                              std::asinh(320.f / 50.f)};
      checkEqual(ticl::TracksterInferenceByShowerShape::features(trackster), expected);
    }
  }

  SECTION("An empty trackster gives finite features") {
    const Features expected{0.f,
                            0.f,
                            0.f,
                            0.f,
                            0.f,
                            kLogMinEigenvalue,
                            kLogMinEigenvalue,
                            kLogMinEigenvalue,
                            1.f / 3.f,
                            0.f,
                            0.f,
                            0.f,
                            0.f,
                            0.f,
                            0.f,
                            0.f};
    checkEqual(ticl::TracksterInferenceByShowerShape::features(ticl::Trackster()), expected);
  }
}

TEST_CASE("Layer-cluster profile of a trackster") {
  using Profile = std::array<float, 6>;
  const Profile noEnergy{-1.f, -1.f, -1.f, -1.f, -1.f, 0.f};

  SECTION("Energy shares, rms and z range around the main axis, for both z signs") {
    for (float s : {1.f, -1.f}) {
      // Distances to the axis 0, 1, 3 and 6 cm; the third layer cluster is shared by 2 tracksters.
      const std::vector<reco::CaloCluster> layerClusters = {layerCluster(0.f, 0.f, 300.f * s, 4.f),
                                                            layerCluster(1.f, 0.f, 310.f * s, 3.f),
                                                            layerCluster(3.f, 0.f, 290.f * s, 2.f),
                                                            layerCluster(0.f, 6.f, 305.f * s, 1.f)};
      auto trackster = tracksterOf(4, 9.f, {0.f, 0.f, 300.f * s}, {1.f, 1.f, 2.f, 1.f});
      setPCA(trackster, Eigen::Vector3f(0.f, 0.f, s));
      const Profile expected{4.f / 9.f, 7.f / 9.f, 8.f / 9.f, std::sqrt(48.f / 9.f), 290.f, 20.f};
      checkEqual(ticl::layerClusterProfile(trackster, layerClusters), expected);
    }
  }

  SECTION("A zero main axis uses the direction of the barycenter") {
    // The barycenter direction is (0, 0.6, 0.8); the second layer cluster is 3 cm from it.
    const std::vector<reco::CaloCluster> layerClusters = {layerCluster(0.f, 231.f, 308.f, 3.f),
                                                          layerCluster(0.f, 227.4f, 298.2f, 1.f)};
    const auto trackster = tracksterOf(2, 4.f, {0.f, 225.f, 300.f});
    const Profile expected{0.75f, 0.75f, 1.f, 1.5f, 298.2f, 9.8f};
    checkEqual(ticl::layerClusterProfile(trackster, layerClusters), expected);
  }

  SECTION("One layer cluster at the barycenter") {
    const std::vector<reco::CaloCluster> layerClusters = {layerCluster(30.f, 40.f, -320.f, 5.f)};
    const auto trackster = tracksterOf(1, 5.f, {30.f, 40.f, -320.f});
    const Profile expected{1.f, 1.f, 1.f, 0.f, 320.f, 0.f};
    checkEqual(ticl::layerClusterProfile(trackster, layerClusters), expected);
  }

  SECTION("Layer clusters without energy") {
    const std::vector<reco::CaloCluster> layerClusters = {layerCluster(0.f, 0.f, 300.f, 0.f),
                                                          layerCluster(1.f, 0.f, 310.f, 0.f)};
    auto trackster = tracksterOf(2, 0.f, {0.5f, 0.f, 305.f});
    setPCA(trackster, Eigen::Vector3f(0.f, 0.f, 1.f));
    checkEqual(ticl::layerClusterProfile(trackster, layerClusters), noEnergy);
  }

  SECTION("An empty trackster") { checkEqual(ticl::layerClusterProfile(ticl::Trackster(), {}), noEnergy); }
}
