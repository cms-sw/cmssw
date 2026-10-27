#include <cassert>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtraHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

// Converts the SoA layer clusters of the HGCAL components into one merged collection of reco::CaloCluster, with their
// time and the layer cluster mask: the layer clusters of the components one after the other, in the order of the
// configuration. It replaces a HGCalLayerClustersFromSoAProducer per component followed by MergeClusterProducer,
// which built every cluster once per component and then copied it twice.
// Layer clusters of other subdetectors (the barrel) can be appended after the HGCAL ones.
class HGCalMergedLayerClustersFromSoAProducer : public edm::global::EDProducer<> {
public:
  explicit HGCalMergedLayerClustersFromSoAProducer(edm::ParameterSet const& config)
      : clustersToken_{produces()},
        maskToken_{produces("InitialLayerClustersMask")},
        timeToken_{produces(config.getParameter<std::string>("timeClname"))} {
    const auto layerClusters = config.getParameter<std::vector<edm::InputTag>>("layerClustersSoA");
    const auto recHits = config.getParameter<std::vector<edm::InputTag>>("hgcalRecHitsSoA");
    const auto clustering = config.getParameter<std::vector<edm::InputTag>>("hgcalRecHitsLayerClustersSoA");
    const auto detectors = config.getParameter<std::vector<std::string>>("detectors");
    if (recHits.size() != layerClusters.size() or clustering.size() != layerClusters.size() or
        detectors.size() != layerClusters.size()) {
      throw cms::Exception("Configuration") << "layerClustersSoA, hgcalRecHitsSoA, hgcalRecHitsLayerClustersSoA and "
                                               "detectors must have one entry per HGCAL component";
    }
    for (unsigned k = 0; k < layerClusters.size(); ++k) {
      components_.push_back({consumes<reco::CaloClusterHostCollection>(layerClusters[k]),
                             consumes<HGCalSoARecHitsHostCollection>(recHits[k]),
                             consumes<HGCalSoARecHitsExtraHostCollection>(clustering[k]),
                             algoId(detectors[k])});
    }

    const auto extra = config.getParameter<std::vector<edm::InputTag>>("extraLayerClusters");
    const auto extraTime = config.getParameter<std::vector<edm::InputTag>>("extraTimeLayerClusters");
    if (extraTime.size() != extra.size()) {
      throw cms::Exception("Configuration") << "extraTimeLayerClusters must have one entry per extraLayerClusters";
    }
    for (unsigned j = 0; j < extra.size(); ++j) {
      extras_.push_back({consumes<std::vector<reco::CaloCluster>>(extra[j]),
                         consumes<edm::ValueMap<std::pair<float, float>>>(extraTime[j])});
    }
  }

  void produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const override {
    // The inputs of every component, read from the event once
    std::vector<Inputs> inputs;
    inputs.reserve(components_.size());
    for (const auto& component : components_) {
      inputs.push_back({iEvent.get(component.layerClusters).const_view(),
                        iEvent.get(component.recHits).const_view(),
                        iEvent.get(component.clustering).const_view()});
    }

    // The layer clusters of component k are rows offsets[k] to offsets[k + 1] of the merged collection
    std::vector<uint32_t> offsets{0};
    for (const auto& input : inputs) {
      offsets.push_back(offsets.back() + input.layerClusters.position().metadata().size());
    }
    const uint32_t nLayerClusters = offsets.back();

    // The hits of every layer cluster, grouped by layer cluster with a counting sort over the rechits: the hits of
    // layer cluster c are hits[hitOffsets[c]] to hits[hitOffsets[c + 1]], in the order of the rechits
    std::vector<uint32_t> hitOffsets(nLayerClusters + 1, 0);
    for (unsigned k = 0; k < inputs.size(); ++k) {
      const auto& clustering = inputs[k].clustering;
      for (int32_t i = 0; i < clustering.metadata().size(); ++i) {
        const int32_t c = clustering[i].clusterIndex();
        if (c >= 0) {
          assert(offsets[k] + c < offsets[k + 1]);
          ++hitOffsets[offsets[k] + c + 1];
        }
      }
    }
    std::partial_sum(hitOffsets.begin(), hitOffsets.end(), hitOffsets.begin());

    std::vector<std::pair<DetId, float>> hits(hitOffsets.back());
    std::vector<uint32_t> next(hitOffsets.begin(), hitOffsets.end() - 1);
    for (unsigned k = 0; k < inputs.size(); ++k) {
      const auto& recHits = inputs[k].recHits;
      const auto& clustering = inputs[k].clustering;
      for (int32_t i = 0; i < clustering.metadata().size(); ++i) {
        const int32_t c = clustering[i].clusterIndex();
        if (c >= 0) {
          hits[next[offsets[k] + c]++] = {DetId(recHits[i].detid()), 1.f};
        }
      }
    }

    std::vector<reco::BasicCluster> clusters;
    std::vector<std::pair<float, float>> times;
    clusters.reserve(nLayerClusters);
    times.reserve(nLayerClusters);
    // Reused for every layer cluster: it grows to the largest one, so the only allocation per layer cluster is the
    // copy that reco::CaloCluster makes of it
    std::vector<std::pair<DetId, float>> clusterHits;
    for (unsigned k = 0; k < inputs.size(); ++k) {
      const auto& view = inputs[k].layerClusters;
      for (uint32_t c = offsets[k]; c < offsets[k + 1]; ++c) {
        const uint32_t i = c - offsets[k];
        clusterHits.assign(hits.begin() + hitOffsets[c], hits.begin() + hitOffsets[c + 1]);
        clusters.emplace_back(view.energy()[i].energy(),
                              math::XYZPoint(view.position()[i].x(), view.position()[i].y(), view.position()[i].z()),
                              reco::CaloID::DET_HGCAL_ENDCAP,
                              clusterHits,
                              components_[k].algoId,
                              view.indexes()[i].seedID());
        times.emplace_back(view.timing()[i].time(), view.timing()[i].timeError());
      }
    }

    for (const auto& extra : extras_) {
      const auto& extraClusters = iEvent.get(extra.clusters);
      const auto& extraTimes = iEvent.get(extra.time);
      clusters.insert(clusters.end(), extraClusters.begin(), extraClusters.end());
      for (size_t i = 0; i < extraClusters.size(); ++i) {
        times.push_back(extraTimes.get(i));
      }
    }

    const auto nClusters = clusters.size();
    const auto clustersHandle = iEvent.emplace(clustersToken_, std::move(clusters));
    iEvent.emplace(maskToken_, nClusters, 1.f);

    edm::ValueMap<std::pair<float, float>> timeMap;
    edm::ValueMap<std::pair<float, float>>::Filler filler(timeMap);
    filler.insert(clustersHandle, times.begin(), times.end());
    filler.fill();
    iEvent.emplace(timeToken_, std::move(timeMap));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<edm::InputTag>>("layerClustersSoA",
                                         {edm::InputTag("hgcalSoALayerClustersEE"),
                                          edm::InputTag("hgcalSoALayerClustersHSi"),
                                          edm::InputTag("hgcalSoALayerClustersHSci")})
        ->setComment("the SoA layer clusters of each HGCAL component, in the order of the merged collection");
    desc.add<std::vector<edm::InputTag>>("hgcalRecHitsSoA",
                                         {edm::InputTag("hgcalSoARecHitsEE"),
                                          edm::InputTag("hgcalSoARecHitsHSi"),
                                          edm::InputTag("hgcalSoARecHitsHSci")})
        ->setComment("the rechits of each component");
    desc.add<std::vector<edm::InputTag>>("hgcalRecHitsLayerClustersSoA",
                                         {edm::InputTag("hgcalCLUEsteringEE"),
                                          edm::InputTag("hgcalCLUEsteringHSi"),
                                          edm::InputTag("hgcalCLUEsteringHSci")})
        ->setComment("the layer cluster of every rechit of each component");
    desc.add<std::vector<std::string>>("detectors", {"EE", "FH", "BH"})
        ->setComment("the HGCAL component of each entry: EE, FH, BH or HFNose");
    desc.add<std::vector<edm::InputTag>>("extraLayerClusters", {})
        ->setComment("legacy layer clusters appended after the HGCAL ones (e.g. the barrel)");
    desc.add<std::vector<edm::InputTag>>("extraTimeLayerClusters", {})
        ->setComment("the time of each extraLayerClusters collection");
    desc.add<std::string>("timeClname", "timeLayerCluster");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  static reco::CaloCluster::AlgoId algoId(const std::string& detector) {
    if (detector == "EE") {
      return reco::CaloCluster::hgcal_em;
    } else if (detector == "FH") {
      return reco::CaloCluster::hgcal_had;
    } else if (detector == "BH") {
      return reco::CaloCluster::hgcal_scintillator;
    } else if (detector == "HFNose") {
      return reco::CaloCluster::hfnose;
    }
    throw cms::Exception("Configuration") << "unknown HGCAL component " << detector;
  }

  struct Component {
    edm::EDGetTokenT<reco::CaloClusterHostCollection> layerClusters;
    edm::EDGetTokenT<HGCalSoARecHitsHostCollection> recHits;
    edm::EDGetTokenT<HGCalSoARecHitsExtraHostCollection> clustering;
    reco::CaloCluster::AlgoId algoId;
  };
  // The views of the inputs of one component in the event being processed
  struct Inputs {
    reco::CaloClusterHostCollection::ConstView layerClusters;
    HGCalSoARecHitsHostCollection::ConstView recHits;
    HGCalSoARecHitsExtraHostCollection::ConstView clustering;
  };
  struct Extra {
    edm::EDGetTokenT<std::vector<reco::CaloCluster>> clusters;
    edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> time;
  };

  std::vector<Component> components_;
  std::vector<Extra> extras_;
  const edm::EDPutTokenT<std::vector<reco::BasicCluster>> clustersToken_;
  const edm::EDPutTokenT<std::vector<float>> maskToken_;
  const edm::EDPutTokenT<edm::ValueMap<std::pair<float, float>>> timeToken_;
};

DEFINE_FWK_MODULE(HGCalMergedLayerClustersFromSoAProducer);
