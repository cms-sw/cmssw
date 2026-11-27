// The shower-shape identification gives probabilities in [0, 1] to the tracksters above minEnergy with the em and the
// mixture models, keeps the probabilities of the other tracksters, and rejects a model that does not match the mode.
// The models come from RecoTICL/Inference/data/showerShapePID in CMSSW_SEARCH_PATH.

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <Eigen/Core>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/TICLGeomTools.h"
#include "RecoTICL/Inference/interface/TICLONNXGlobalCache.h"
#include "RecoTICL/Inference/interface/TracksterInferenceByShowerShape.h"

namespace {

  const std::string kModelDir = "RecoTICL/Inference/data/showerShapePID/";
  const std::string kEmModel = kModelDir + "shapePID_v1.onnx";
  const std::string kMixtureModel = kModelDir + "mixPID_hi_v2.onnx";
  constexpr float kMinEnergy = 1.f;

  // The ONNXService gives the ONNX Runtime environment of the sessions while the token is alive.
  edm::OwningServiceToken makeServices() {
    if (not edmplugin::PluginManager::isAvailable())
      edmplugin::PluginManager::configure(edmplugin::standard::config());
    std::vector<edm::ParameterSet> psets;
    for (const char* service : {"ResourceInformationService", "ONNXService"}) {
      edm::ParameterSet pset;
      pset.addParameter<std::string>("@service_type", service);
      psets.push_back(pset);
    }
    return edm::ServiceRegistry::createSet(psets);
  }

  edm::ParameterSet pluginPSet(const std::string& mode,
                               const std::string& model,
                               const std::vector<std::string>& outputNames,
                               const std::vector<std::string>& inputNames = {"input"}) {
    edm::ParameterSetDescription desc;
    ticl::TracksterInferenceByShowerShape::fillPSetDescription(desc);
    edm::ParameterSet pset;
    pset.addParameter<std::string>("mode", mode);
    pset.addParameter<std::string>("onnxModelPath", model);
    pset.addParameter<std::vector<std::string>>("inputNames", inputNames);
    pset.addParameter<std::vector<std::string>>("outputNames", outputNames);
    pset.addParameter<double>("minEnergy", kMinEnergy);
    desc.validate(pset);
    return pset;
  }

  std::unique_ptr<ticl::TICLONNXGlobalCache> cacheOf(const edm::ParameterSet& plugin) {
    edm::ParameterSet module;
    module.addParameter<std::string>("inferenceAlgo", "TracksterInferenceByShowerShape");
    module.addParameter<edm::ParameterSet>("pluginInferenceAlgoTracksterInferenceByShowerShape", plugin);
    return ticl::TICLONNXGlobalCache::initialize(module);
  }

  // The category of the cms::Exception that the construction of the plugin throws, or "" without exception.
  std::string constructionError(const edm::ParameterSet& plugin) {
    const auto cache = cacheOf(plugin);
    try {
      ticl::TracksterInferenceByShowerShape algo(plugin, cache.get());
    } catch (cms::Exception const& e) {
      return e.category();
    }
    return "";
  }

  reco::CaloCluster layerCluster(float x, float y, float z, float energy) {
    return reco::CaloCluster(
        energy, math::XYZPoint(x, y, z), reco::CaloID(reco::CaloID::DET_HGCAL_ENDCAP), reco::CaloCluster::hgcal_em);
  }

  // Ten layer clusters near (30, 40) at z from 322 to 340 cm, then the same ten at negative z.
  std::vector<reco::CaloCluster> layerClusters() {
    std::vector<reco::CaloCluster> clusters;
    for (float zSign : {1.f, -1.f})
      for (int i = 0; i < 10; ++i)
        clusters.push_back(layerCluster(30.f + 0.2f * i, 40.f, zSign * (322.f + 2.f * i), 2.f - 0.15f * i));
    return clusters;
  }

  // A trackster of the layer clusters [first, first + n); with `pca`, its main axis is the barycenter direction.
  ticl::Trackster tracksterOf(const std::vector<reco::CaloCluster>& clusters, unsigned first, unsigned n, bool pca) {
    ticl::Trackster trackster;
    float energy = 0.f;
    Eigen::Vector3f barycenter = Eigen::Vector3f::Zero();
    for (unsigned i = first; i < first + n; ++i) {
      trackster.vertices().push_back(i);
      trackster.vertex_multiplicity().push_back(1.f);
      const auto e = static_cast<float>(clusters[i].energy());
      energy += e;
      barycenter += e * Eigen::Vector3f(clusters[i].x(), clusters[i].y(), clusters[i].z());
    }
    barycenter /= energy;
    trackster.setRawEnergy(energy);
    trackster.setRawEmEnergy(0.8f * energy);
    trackster.setBarycenter({barycenter.x(), barycenter.y(), barycenter.z()});
    if (pca) {
      Eigen::Matrix3f eigenvectors;
      eigenvectors.setZero();
      eigenvectors.col(0) = barycenter.normalized();
      trackster.fillPCAVariables(Eigen::Vector3f(30.f, 2.f, 1.f),
                                 eigenvectors,
                                 Eigen::Vector3f(4.f, 4.f, 30.f),
                                 Eigen::Vector3f(30.f, 2.f, 1.f),
                                 3,
                                 ticl::Trackster::PCAOrdering::descending);
    }
    return trackster;
  }

  // Tracksters at positive and negative z, one without PCA, and the last one below minEnergy with P(unknown) = 1.
  std::vector<ticl::Trackster> tracksters(const std::vector<reco::CaloCluster>& clusters) {
    std::vector<ticl::Trackster> result{tracksterOf(clusters, 0, 10, true),
                                        tracksterOf(clusters, 10, 10, true),
                                        tracksterOf(clusters, 0, 5, false),
                                        tracksterOf(clusters, 19, 1, false)};
    REQUIRE(result.back().raw_energy() < kMinEnergy);
    result.back().setProbabilitiesUnknown();
    return result;
  }

  float sumOf(const ticl::Trackster& trackster) {
    const auto& p = trackster.id_probabilities();
    return std::accumulate(p.begin(), p.end(), 0.f);
  }

}  // namespace

TEST_CASE("Shower-shape identification with the ONNX models") {
  const auto token = makeServices();
  edm::ServiceRegistry::Operate operate(token);
  const ticlgeom::Tools geometry;
  const auto clusters = layerClusters();

  SECTION("Mode em: P(photon) in [0, 1] and P(charged hadron) = 1 - P(photon)") {
    const auto plugin = pluginPSet("em", kEmModel, {"em"});
    const auto cache = cacheOf(plugin);
    const ticl::TracksterInferenceByShowerShape algo(plugin, cache.get());
    auto result = tracksters(clusters);
    algo.runInference(clusters, result, geometry);
    for (std::size_t i = 0; i + 1 < result.size(); ++i) {
      INFO("trackster " << i);
      const float photon = result[i].id_probability(ticl::Trackster::ParticleType::photon);
      CHECK(photon >= 0.f);
      CHECK(photon <= 1.f);
      CHECK(result[i].id_probability(ticl::Trackster::ParticleType::charged_hadron) == Catch::Approx(1.f - photon));
      CHECK(sumOf(result[i]) == Catch::Approx(1.f));
    }
    CHECK(result.back().id_probability(ticl::Trackster::ParticleType::unknown) == 1.f);
    CHECK(sumOf(result.back()) == 1.f);
  }

  SECTION("Mode mixture: 8 probabilities in [0, 1]") {
    const auto plugin = pluginPSet("mixture", kMixtureModel, {"pid"});
    const auto cache = cacheOf(plugin);
    const ticl::TracksterInferenceByShowerShape algo(plugin, cache.get());
    auto result = tracksters(clusters);
    algo.runInference(clusters, result, geometry);
    for (std::size_t i = 0; i + 1 < result.size(); ++i) {
      INFO("trackster " << i);
      for (float p : result[i].id_probabilities()) {
        CHECK(p >= 0.f);
        CHECK(p <= 1.f);
      }
      CHECK(sumOf(result[i]) > 0.f);
    }
    CHECK(result.back().id_probability(ticl::Trackster::ParticleType::unknown) == 1.f);
    CHECK(sumOf(result.back()) == 1.f);
  }

  SECTION("A model that does not match the mode is a configuration error") {
    CHECK(constructionError(pluginPSet("em", kEmModel, {"em"})).empty());
    CHECK(constructionError(pluginPSet("mixture", kMixtureModel, {"pid"})).empty());
    CHECK(constructionError(pluginPSet("mixture", kMixtureModel, {"em"})) == "Configuration");
    CHECK(constructionError(pluginPSet("mixture", kEmModel, {"em"})) == "Configuration");
    CHECK(constructionError(pluginPSet("em", kMixtureModel, {"pid"})) == "Configuration");
    CHECK(constructionError(pluginPSet("em", kEmModel, {"em", "pid"})) == "Configuration");
    CHECK(constructionError(pluginPSet("em", kEmModel, {"em"}, {"x"})) == "Configuration");
  }
}
