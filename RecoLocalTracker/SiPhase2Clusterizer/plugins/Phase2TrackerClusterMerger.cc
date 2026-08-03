#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Phase2TrackerClusterMergerAlgorithm.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"

#include <memory>

/*
 * Second stage clusterizer: the front end closes a cluster once it reaches the hardware
 * size limit, so a single physical cluster can be read out as several clusters whose
 * edges sit on adjacent digi channels. This producer merges those back together.
 */

class Phase2TrackerClusterMerger : public edm::stream::EDProducer<> {
public:
  explicit Phase2TrackerClusterMerger(const edm::ParameterSet& conf);
  ~Phase2TrackerClusterMerger() override = default;
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> const token_;
};

Phase2TrackerClusterMerger::Phase2TrackerClusterMerger(edm::ParameterSet const& conf)
    : token_(consumes<Phase2TrackerCluster1DCollectionNew>(conf.getParameter<edm::InputTag>("src"))) {
  produces<Phase2TrackerCluster1DCollectionNew>();
}

void Phase2TrackerClusterMerger::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // Get the Clusters
  edm::Handle<Phase2TrackerCluster1DCollectionNew> inputClusters;
  event.getByToken(token_, inputClusters);

  // Global container for the merged clusters of each module
  auto outputClusters = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

  // Merging never reaches across modules, so each DetSet is handled on its own. No
  // geometry is needed: adjacency is decided from the packed digi channel alone.
  for (const auto& DSViter : *inputClusters) {
    Phase2TrackerCluster1DCollectionNew::FastFiller clusters(*outputClusters, DSViter.detId());
    Phase2TrackerClusterMergerAlgorithm algo;
    algo.mergeDetUnit(DSViter, clusters);
    if (clusters.empty())
      clusters.abort();
  }

  // Add the data to the output
  outputClusters->shrink_to_fit();
  event.put(std::move(outputClusters));
}

void Phase2TrackerClusterMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("siPhase2Clusters"));
  descriptions.add("default_phase2TrackerClusterMerger", desc);
}

DEFINE_FWK_MODULE(Phase2TrackerClusterMerger);
