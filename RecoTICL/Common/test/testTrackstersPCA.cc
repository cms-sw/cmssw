// The principal components of a trackster whose PCA set has no spread to normalise (one layer cluster carries all
// its energy) are all zeros, and the ones of a regular trackster are finite.

#include <cmath>
#include <utility>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Common/interface/TrackstersPCA.h"

namespace {

  constexpr double kZLimitEm = 400.;

  reco::CaloCluster layerCluster(float x, float y, float z, float energy) {
    return reco::CaloCluster(
        energy, math::XYZPoint(x, y, z), reco::CaloID(reco::CaloID::DET_HGCAL_ENDCAP), reco::CaloCluster::hgcal_em);
  }

  // One trackster of all the layer clusters, with the PCA of ticl::assignPCAtoTracksters without the cleaning
  ticl::Trackster tracksterOf(const std::vector<reco::CaloCluster>& layerClusters) {
    std::vector<std::pair<float, float>> times(layerClusters.size(), {-99.f, -1.f});
    edm::ValueMap<std::pair<float, float>> layerClustersTime;
    edm::ValueMap<std::pair<float, float>>::Filler filler(layerClustersTime);
    edm::TestHandle<std::vector<reco::CaloCluster>> handle(&layerClusters, edm::ProductID(1, 1));
    filler.insert(handle, times.begin(), times.end());
    filler.fill();

    std::vector<ticl::Trackster> tracksters(1);
    for (unsigned i = 0; i < layerClusters.size(); ++i) {
      tracksters[0].vertices().push_back(i);
      tracksters[0].vertex_multiplicity().push_back(1.f);
    }
    const ticlgeom::Tools geometry;
    ticl::assignPCAtoTracksters(tracksters, layerClusters, layerClustersTime, kZLimitEm, geometry, true, true, false);
    return tracksters[0];
  }

}  // namespace

TEST_CASE("PCA of a trackster") {
  SECTION("One layer cluster with all the energy gives a PCA of zeros") {
    const std::vector<reco::CaloCluster> layerClusters = {layerCluster(50.f, 10.f, 330.f, 5.f),
                                                          layerCluster(51.f, 10.f, 331.f, 0.f),
                                                          layerCluster(50.f, 11.f, 332.f, 0.f)};
    const auto trackster = tracksterOf(layerClusters);
    CHECK(trackster.raw_energy() == Catch::Approx(5.f));
    for (int k = 0; k < 3; ++k) {
      CHECK(trackster.eigenvalues()[k] == 0.f);
      CHECK(trackster.sigmas()[k] == 0.f);
      CHECK(trackster.sigmasPCA()[k] == 0.f);
      CHECK(trackster.eigenvectors(k).x() == 0.f);
      CHECK(trackster.eigenvectors(k).y() == 0.f);
      CHECK(trackster.eigenvectors(k).z() == 0.f);
    }
  }

  SECTION("Three layer clusters along a line give a finite PCA with the principal axis along the line") {
    const std::vector<reco::CaloCluster> layerClusters = {layerCluster(60.f, 20.f, 340.f, 1.f),
                                                          layerCluster(60.5f, 20.2f, 341.f, 1.f),
                                                          layerCluster(61.f, 20.4f, 342.f, 1.f)};
    const auto trackster = tracksterOf(layerClusters);
    for (int k = 0; k < 3; ++k) {
      CHECK(std::isfinite(trackster.eigenvalues()[k]));
      CHECK(std::isfinite(trackster.sigmas()[k]));
      CHECK(std::isfinite(trackster.sigmasPCA()[k]));
    }
    CHECK(trackster.eigenvalues()[0] > 0.f);
    // the line has direction (0.5, 0.2, 1) per layer; the principal axis points away from the interaction point
    const float norm = std::sqrt(0.5f * 0.5f + 0.2f * 0.2f + 1.f);
    CHECK(trackster.eigenvectors(0).x() == Catch::Approx(0.5f / norm).margin(1e-3));
    CHECK(trackster.eigenvectors(0).y() == Catch::Approx(0.2f / norm).margin(1e-3));
    CHECK(trackster.eigenvectors(0).z() == Catch::Approx(1.f / norm).margin(1e-3));
  }
}
