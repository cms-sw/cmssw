#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtraHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

namespace edmtest {

  // Puts the rechit SoA of the configured event (event e uses events[(e - 1) % size]) and the number of rechits of
  // each layer, plus layerSizesExcess on the last layer. The rechits of an event must be grouped by layer.
  class TestHGCalSoARecHitsSource : public edm::global::EDProducer<> {
  public:
    explicit TestHGCalSoARecHitsSource(edm::ParameterSet const& config)
        : events_(config.getParameter<std::vector<edm::ParameterSet>>("events")),
          layerSizesExcess_(config.getParameter<unsigned int>("layerSizesExcess")),
          recHitsToken_(produces()),
          layerSizesToken_(produces("layerSizes")) {}

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
      const auto& hits = events_[(event.id().event() - 1) % events_.size()];
      const auto dim1 = hits.getParameter<std::vector<double>>("dim1");
      const auto dim2 = hits.getParameter<std::vector<double>>("dim2");
      const auto dim3 = hits.getParameter<std::vector<double>>("dim3");
      const auto energy = hits.getParameter<std::vector<double>>("energy");
      const auto sigmaNoise = hits.getParameter<std::vector<double>>("sigmaNoise");
      const auto time = hits.getParameter<std::vector<double>>("time");
      const auto timeError = hits.getParameter<std::vector<double>>("timeError");
      const auto layer = hits.getParameter<std::vector<int>>("layer");
      const auto detid = hits.getParameter<std::vector<unsigned int>>("detid");
      const auto n = static_cast<int32_t>(energy.size());
      for (const auto size : {dim1.size(),
                              dim2.size(),
                              dim3.size(),
                              sigmaNoise.size(),
                              time.size(),
                              timeError.size(),
                              layer.size(),
                              detid.size()}) {
        if (size != energy.size()) {
          throw cms::Exception("Configuration") << "all the columns of an event must have the same size";
        }
      }

      HGCalSoARecHitsHostCollection recHits(n);
      auto view = recHits.view();
      std::vector<uint32_t> layerSizes;
      for (int32_t i = 0; i < n; ++i) {
        view[i].dim1() = dim1[i];
        view[i].dim2() = dim2[i];
        view[i].dim3() = dim3[i];
        view[i].energy() = energy[i];
        view[i].mipEnergy() = energy[i];
        view[i].sigmaNoise() = sigmaNoise[i];
        view[i].time() = time[i];
        view[i].timeError() = timeError[i];
        view[i].layer() = layer[i];
        view[i].detid() = detid[i];
        view[i].recHitIndex() = i;
        view[i].flags() = 0;
        if (i == 0 or layer[i] != layer[i - 1]) {
          layerSizes.push_back(0);
        }
        ++layerSizes.back();
      }
      if (layerSizesExcess_ > 0) {
        if (layerSizes.empty()) {
          layerSizes.push_back(0);
        }
        layerSizes.back() += layerSizesExcess_;
      }
      event.emplace(recHitsToken_, std::move(recHits));
      event.emplace(layerSizesToken_, std::move(layerSizes));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription hits;
      hits.add<std::vector<double>>("dim1", {});
      hits.add<std::vector<double>>("dim2", {});
      hits.add<std::vector<double>>("dim3", {});
      hits.add<std::vector<double>>("energy", {});
      hits.add<std::vector<double>>("sigmaNoise", {});
      hits.add<std::vector<double>>("time", {});
      hits.add<std::vector<double>>("timeError", {});
      hits.add<std::vector<int>>("layer", {});
      hits.add<std::vector<unsigned int>>("detid", {});
      edm::ParameterSetDescription desc;
      desc.addVPSet("events", hits, {});
      desc.add<unsigned int>("layerSizesExcess", 0);
      descriptions.addDefault(desc);
    }

  private:
    const std::vector<edm::ParameterSet> events_;
    const unsigned int layerSizesExcess_;
    const edm::EDPutTokenT<HGCalSoARecHitsHostCollection> recHitsToken_;
    const edm::EDPutTokenT<std::vector<uint32_t>> layerSizesToken_;
  };

  // Puts n legacy layer clusters and their times, with n the event number, as the layer clusters of the barrel
  class TestLegacyLayerClustersSource : public edm::global::EDProducer<> {
  public:
    explicit TestLegacyLayerClustersSource(edm::ParameterSet const&)
        : clustersToken_(produces()), timeToken_(produces("timeLayerCluster")) {}

    void produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
      const auto n = event.id().event();
      std::vector<reco::BasicCluster> clusters;
      std::vector<std::pair<float, float>> times;
      for (unsigned int i = 0; i < n; ++i) {
        clusters.emplace_back(1.f + i,
                              math::XYZPoint(10.f * i, 0.f, 0.f),
                              reco::CaloID::DET_HGCAL_ENDCAP,
                              std::vector<std::pair<DetId, float>>{{DetId(i + 1), 1.f}},
                              reco::CaloCluster::barrel_em);
        times.emplace_back(0.1f * i, 0.01f);
      }
      const auto handle = event.emplace(clustersToken_, std::move(clusters));
      edm::ValueMap<std::pair<float, float>> timeMap;
      edm::ValueMap<std::pair<float, float>>::Filler filler(timeMap);
      filler.insert(handle, times.begin(), times.end());
      filler.fill();
      event.emplace(timeToken_, std::move(timeMap));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      descriptions.addDefault(edm::ParameterSetDescription());
    }

  private:
    const edm::EDPutTokenT<std::vector<reco::BasicCluster>> clustersToken_;
    const edm::EDPutTokenT<edm::ValueMap<std::pair<float, float>>> timeToken_;
  };

  // Checks the layer clusters of the HGCAL components against their rechits and clustering: one seed per layer
  // cluster, their energy, number of cells, seed, algorithm, layer, position and time. Then checks the legacy merged
  // layer clusters against the ones of the components.
  class TestHGCalLayerClustersChecker : public edm::global::EDAnalyzer<> {
  public:
    explicit TestHGCalLayerClustersChecker(edm::ParameterSet const& config)
        : mergedToken_(consumes(config.getParameter<edm::InputTag>("mergedLayerClusters"))),
          maskToken_(consumes(edm::InputTag(config.getParameter<edm::InputTag>("mergedLayerClusters").label(),
                                            "InitialLayerClustersMask"))),
          timeToken_(consumes(
              edm::InputTag(config.getParameter<edm::InputTag>("mergedLayerClusters").label(), "timeLayerCluster"))),
          extraToken_(consumes(config.getParameter<edm::InputTag>("extraLayerClusters"))),
          extraTimeToken_(consumes(
              edm::InputTag(config.getParameter<edm::InputTag>("extraLayerClusters").label(), "timeLayerCluster"))) {
      for (const auto& pset : config.getParameter<std::vector<edm::ParameterSet>>("components")) {
        components_.push_back(
            {consumes<HGCalSoARecHitsHostCollection>(pset.getParameter<edm::InputTag>("recHits")),
             consumes<HGCalSoARecHitsExtraHostCollection>(pset.getParameter<edm::InputTag>("clustering")),
             consumes<reco::CaloClusterHostCollection>(pset.getParameter<edm::InputTag>("layerClusters")),
             static_cast<reco::CaloCluster::AlgoId>(pset.getParameter<int>("algoId")),
             pset.getParameter<bool>("isScintillator"),
             pset.getParameter<std::vector<int>>("expectedNumberOfClusters")});
      }
    }

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
      const auto eventNumber = event.id().event();
      std::vector<reco::CaloClusterHostCollection::ConstView> componentClusters;
      std::vector<std::vector<std::vector<int32_t>>> componentHits;
      std::vector<HGCalSoARecHitsHostCollection::ConstView> componentRecHits;
      for (const auto& component : components_) {
        const auto recHits = event.get(component.recHits).const_view();
        const auto clustering = event.get(component.clustering).const_view();
        const auto clusters = event.get(component.layerClusters).const_view();
        const auto hits = checkClustering(eventNumber, component, recHits, clustering);
        checkLayerClusters(component, recHits, hits, clustering, clusters);
        componentClusters.push_back(clusters);
        componentHits.push_back(hits);
        componentRecHits.push_back(recHits);
      }
      checkMergedLegacy(componentClusters,
                        componentHits,
                        componentRecHits,
                        event.get(extraToken_),
                        event.get(extraTimeToken_),
                        event.get(mergedToken_),
                        event.get(maskToken_),
                        event.get(timeToken_));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription component;
      component.add<edm::InputTag>("recHits");
      component.add<edm::InputTag>("clustering");
      component.add<edm::InputTag>("layerClusters");
      component.add<int>("algoId");
      component.add<bool>("isScintillator", false);
      component.add<std::vector<int>>("expectedNumberOfClusters", {})
          ->setComment("per event, from the first one; a negative number is not checked");
      edm::ParameterSetDescription desc;
      desc.addVPSet("components", component, {});
      desc.add<edm::InputTag>("mergedLayerClusters");
      desc.add<edm::InputTag>("extraLayerClusters");
      descriptions.addDefault(desc);
    }

  private:
    struct Component {
      edm::EDGetTokenT<HGCalSoARecHitsHostCollection> recHits;
      edm::EDGetTokenT<HGCalSoARecHitsExtraHostCollection> clustering;
      edm::EDGetTokenT<reco::CaloClusterHostCollection> layerClusters;
      reco::CaloCluster::AlgoId algoId;
      bool isScintillator;
      std::vector<int> expectedNumberOfClusters;
    };

    static void require(bool condition, const std::string& what) {
      if (not condition) {
        throw cms::Exception("TestFailed") << what;
      }
    }

    static bool close(float a, float b, float tolerance = 1e-5f) {
      return std::abs(a - b) <= tolerance * std::max({1.f, std::abs(a), std::abs(b)});
    }

    // The rechits of each layer cluster, in rechit order, after the check of one seed per layer cluster
    static std::vector<std::vector<int32_t>> checkClustering(unsigned int eventNumber,
                                                             const Component& component,
                                                             HGCalSoARecHitsHostCollection::ConstView recHits,
                                                             HGCalSoARecHitsExtraHostCollection::ConstView clustering) {
      const int32_t nRecHits = recHits.metadata().size();
      require(clustering.metadata().size() == nRecHits, "one clustering row per rechit");
      const int32_t nClusters = clustering.numberOfClustersScalar();
      if (eventNumber <= component.expectedNumberOfClusters.size()) {
        const int expected = component.expectedNumberOfClusters[eventNumber - 1];
        require(expected < 0 or expected == nClusters,
                "event " + std::to_string(eventNumber) + ": " + std::to_string(nClusters) +
                    " layer clusters instead of " + std::to_string(expected));
      }
      std::vector<std::vector<int32_t>> hits(nClusters);
      std::vector<int32_t> seeds(nClusters, 0);
      for (int32_t i = 0; i < nRecHits; ++i) {
        const int32_t c = clustering[i].clusterIndex();
        require(c >= -1 and c < nClusters, "layer cluster index out of range");
        if (clustering[i].isSeed()) {
          require(c >= 0, "a seed without a layer cluster");
          ++seeds[c];
        }
        if (c >= 0) {
          hits[c].push_back(i);
        }
      }
      require(std::all_of(seeds.begin(), seeds.end(), [](int32_t n) { return n == 1; }), "one seed per layer cluster");
      return hits;
    }

    static void checkLayerClusters(const Component& component,
                                   HGCalSoARecHitsHostCollection::ConstView recHits,
                                   const std::vector<std::vector<int32_t>>& hits,
                                   HGCalSoARecHitsExtraHostCollection::ConstView clustering,
                                   reco::CaloClusterHostCollection::ConstView clusters) {
      constexpr float thresholdW0 = 2.9f;
      constexpr float positionDeltaRho2 = 1.69f;
      constexpr unsigned int nHitsTime = 3;
      const int32_t nClusters = hits.size();
      require(clusters.position().metadata().size() == nClusters, "one SoA layer cluster per layer cluster");
      for (int32_t c = 0; c < nClusters; ++c) {
        float energy = 0.f;
        int32_t maxHit = -1;
        uint32_t seed = 0;
        std::vector<float> times;
        std::vector<float> weights;
        for (const int32_t i : hits[c]) {
          energy += recHits[i].energy();
          if (maxHit < 0 or recHits[i].energy() > recHits[maxHit].energy()) {
            maxHit = i;
          }
          if (clustering[i].isSeed()) {
            seed = recHits[i].detid();
          }
          if (not component.isScintillator and recHits[i].timeError() >= 0.f) {
            times.push_back(recHits[i].time());
            weights.push_back(1.f / (recHits[i].timeError() * recHits[i].timeError()));
          }
        }
        // the position: energy weights for the scintillator, log weights around the most energetic rechit for silicon
        float x = 0.f;
        float y = 0.f;
        float weightSum = 0.f;
        for (const int32_t i : hits[c]) {
          float hx = recHits[i].dim1();
          float hy = recHits[i].dim2();
          float weight = recHits[i].energy();
          if (component.isScintillator) {
            const float r = recHits[i].dim3() / std::sinh(recHits[i].dim1());
            hx = r * std::cos(recHits[i].dim2());
            hy = r * std::sin(recHits[i].dim2());
          } else {
            const float d1 = recHits[i].dim1() - recHits[maxHit].dim1();
            const float d2 = recHits[i].dim2() - recHits[maxHit].dim2();
            if (d1 * d1 + d2 * d2 > positionDeltaRho2) {
              continue;
            }
            weight = std::max(thresholdW0 + std::log(recHits[i].energy() / energy), 0.f);
          }
          x += hx * weight;
          y += hy * weight;
          weightSum += weight;
        }
        require(weightSum > 0.f, "the layer cluster position has no weight");
        hgcalsimclustertime::ComputeClusterTime timeEstimator;
        const auto time = timeEstimator.fixSizeHighestDensity(times, weights, nHitsTime);

        const std::string which = "layer cluster " + std::to_string(c) + ": ";
        require(close(clusters.energy()[c].energy(), energy), which + "energy");
        require(clusters.position()[c].cells() == static_cast<int>(hits[c].size()), which + "number of cells");
        require(clusters.indexes()[c].seedID() == seed, which + "seed");
        require(clusters.indexes()[c].algoID() == component.algoId, which + "algorithm");
        require(clusters.position()[c].layer() == recHits[maxHit].layer(), which + "layer");
        require(clusters.position()[c].z() == recHits[maxHit].dim3(), which + "z");
        require(close(clusters.position()[c].x(), x / weightSum, 1e-4f), which + "x");
        require(close(clusters.position()[c].y(), y / weightSum, 1e-4f), which + "y");
        require(close(clusters.timing()[c].time(), time.first), which + "time");
        require(close(clusters.timing()[c].timeError(), time.second), which + "time error");
      }
    }

    // The legacy merged layer clusters are the ones of the components, then the extra ones, with their hits and times
    static void checkMergedLegacy(const std::vector<reco::CaloClusterHostCollection::ConstView>& components,
                                  const std::vector<std::vector<std::vector<int32_t>>>& componentHits,
                                  const std::vector<HGCalSoARecHitsHostCollection::ConstView>& componentRecHits,
                                  const std::vector<reco::BasicCluster>& extra,
                                  const edm::ValueMap<std::pair<float, float>>& extraTime,
                                  const std::vector<reco::BasicCluster>& merged,
                                  const std::vector<float>& mask,
                                  const edm::ValueMap<std::pair<float, float>>& time) {
      size_t row = 0;
      for (unsigned k = 0; k < components.size(); ++k) {
        const auto& clusters = components[k];
        for (int32_t c = 0; c < clusters.position().metadata().size(); ++c, ++row) {
          require(row < merged.size(), "merged legacy: too few layer clusters");
          const auto& cluster = merged[row];
          const std::string which = "merged legacy layer cluster " + std::to_string(row) + ": ";
          require(cluster.energy() == clusters.energy()[c].energy(), which + "energy");
          require(cluster.position().x() == clusters.position()[c].x() and
                      cluster.position().y() == clusters.position()[c].y() and
                      cluster.position().z() == clusters.position()[c].z(),
                  which + "position");
          require(cluster.algo() == clusters.indexes()[c].algoID(), which + "algorithm");
          require(cluster.seed().rawId() == clusters.indexes()[c].seedID(), which + "seed");
          const auto& hitsAndFractions = cluster.hitsAndFractions();
          require(hitsAndFractions.size() == componentHits[k][c].size(), which + "number of hits");
          for (size_t h = 0; h < hitsAndFractions.size(); ++h) {
            require(hitsAndFractions[h].first.rawId() == componentRecHits[k][componentHits[k][c][h]].detid() and
                        hitsAndFractions[h].second == 1.f,
                    which + "hits");
          }
          require(time.get(row).first == clusters.timing()[c].time() and
                      time.get(row).second == clusters.timing()[c].timeError(),
                  which + "time");
        }
      }
      for (size_t e = 0; e < extra.size(); ++e, ++row) {
        require(row < merged.size(), "merged legacy: too few layer clusters");
        require(merged[row].energy() == extra[e].energy() and merged[row].algo() == extra[e].algo(),
                "merged legacy: extra layer cluster");
        require(time.get(row) == extraTime.get(e), "merged legacy: time of an extra layer cluster");
      }
      require(row == merged.size(), "merged legacy: too many layer clusters");
      require(mask.size() == merged.size() and std::all_of(mask.begin(), mask.end(), [](float m) { return m == 1.f; }),
              "merged legacy: mask");
    }

    std::vector<Component> components_;
    const edm::EDGetTokenT<std::vector<reco::BasicCluster>> mergedToken_;
    const edm::EDGetTokenT<std::vector<float>> maskToken_;
    const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> timeToken_;
    const edm::EDGetTokenT<std::vector<reco::BasicCluster>> extraToken_;
    const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> extraTimeToken_;
  };

}  // namespace edmtest

using edmtest::TestHGCalLayerClustersChecker;
using edmtest::TestHGCalSoARecHitsSource;
using edmtest::TestLegacyLayerClustersSource;
DEFINE_FWK_MODULE(TestHGCalSoARecHitsSource);
DEFINE_FWK_MODULE(TestLegacyLayerClustersSource);
DEFINE_FWK_MODULE(TestHGCalLayerClustersChecker);
