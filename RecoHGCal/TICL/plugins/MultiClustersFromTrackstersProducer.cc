// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

class MultiClustersFromTrackstersProducer : public edm::stream::EDProducer<> {
public:
  MultiClustersFromTrackstersProducer(const edm::ParameterSet&);
  ~MultiClustersFromTrackstersProducer() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> layer_clusters_token_;
  edm::EDGetTokenT<std::vector<ticl::Trackster>> tracksters_token_;
};

DEFINE_FWK_MODULE(MultiClustersFromTrackstersProducer);

MultiClustersFromTrackstersProducer::MultiClustersFromTrackstersProducer(const edm::ParameterSet& ps)
    : layer_clusters_token_(consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("LayerClusters"))),
      tracksters_token_(consumes<std::vector<ticl::Trackster>>(ps.getParameter<edm::InputTag>("Tracksters"))) {
  produces<std::vector<reco::HGCalMultiCluster>>();
}

void MultiClustersFromTrackstersProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMultiClusters
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("Tracksters", edm::InputTag("Tracksters", "TrackstersByCA"));
  desc.add<edm::InputTag>("LayerClusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.addUntracked<unsigned int>("verbosity", 3);
  descriptions.add("multiClustersFromTrackstersProducer", desc);
}

void MultiClustersFromTrackstersProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  auto multiclusters = std::make_unique<std::vector<reco::HGCalMultiCluster>>();
  edm::Handle<std::vector<ticl::Trackster>> tracksterHandle;
  evt.getByToken(tracksters_token_, tracksterHandle);

  edm::Handle<std::vector<reco::CaloCluster>> layer_clustersHandle;
  evt.getByToken(layer_clusters_token_, layer_clustersHandle);

  auto const& tracksters = *tracksterHandle;
  auto const& layerClusters = *layer_clustersHandle;

  edm::PtrVector<reco::BasicCluster> clusterPtrs;
  for (unsigned i = 0; i < layerClusters.size(); ++i) {
    edm::Ptr<reco::BasicCluster> ptr(layer_clustersHandle, i);
    clusterPtrs.push_back(ptr);
  }

  std::for_each(std::begin(tracksters), std::end(tracksters), [&](auto const& trackster) {
    // Create an empty multicluster if the trackster has no layer clusters.
    // This could happen when a seed leads to no trackster and a dummy one is produced.

    std::array<double, 3> baricenter{{0., 0., 0.}};
    double total_weight = 0.;
    reco::HGCalMultiCluster temp;
    int counter = 0;
    if (!trackster.vertices().empty()) {
      std::for_each(std::begin(trackster.vertices()), std::end(trackster.vertices()), [&](unsigned int idx) {
        temp.push_back(clusterPtrs[idx]);
        auto fraction = 1.f / trackster.vertex_multiplicity(counter++);
        for (auto const& cell : clusterPtrs[idx]->hitsAndFractions()) {
          temp.addHitAndFraction(cell.first, cell.second * fraction);
        }
        auto weight = clusterPtrs[idx]->energy() * fraction;
        total_weight += weight;
        baricenter[0] += clusterPtrs[idx]->x() * weight;
        baricenter[1] += clusterPtrs[idx]->y() * weight;
        baricenter[2] += clusterPtrs[idx]->z() * weight;
      });
      std::transform(
          std::begin(baricenter), std::end(baricenter), std::begin(baricenter), [&total_weight](double val) -> double {
            return val / total_weight;
          });
    }
    temp.setEnergy(total_weight);
    temp.setCorrectedEnergy(total_weight);
    temp.setPosition(math::XYZPoint(baricenter[0], baricenter[1], baricenter[2]));
    temp.setAlgoId(reco::CaloCluster::hgcal_em);
    temp.setTime(trackster.time(), trackster.timeError());
    multiclusters->push_back(temp);
  });

  evt.put(std::move(multiclusters));
}
