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

#include "CondFormats/DataRecord/interface/SiPhase2OuterTrackerCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripBadStrip.h"

/*
 * Second stage clusterizer: the front end closes a cluster once it reaches the hardware
 * size limit, so a single physical cluster can be read out as several clusters whose
 * edges sit on adjacent digi channels. This producer merges those back together.
 * 
 * Additionally, it merges clusters that are separated by a user-defined number
 * of bad strips
 */

class Phase2TrackerClusterMerger : public edm::stream::EDProducer<> {
public:
  explicit Phase2TrackerClusterMerger(const edm::ParameterSet& conf);
  ~Phase2TrackerClusterMerger() override = default;
  void produce(edm::Event& event, const edm::EventSetup& eventSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> const token_;

  // bad strip access
  const bool handleBadStrips_;
  const int maxBadStrips_;
  // <type of object from the payload, record name>
  edm::ESGetToken<SiStripBadStrip, SiPhase2OuterTrackerBadStripRcd> badStripToken_;

};

Phase2TrackerClusterMerger::Phase2TrackerClusterMerger(edm::ParameterSet const& conf)
    : token_(consumes<Phase2TrackerCluster1DCollectionNew>(conf.getParameter<edm::InputTag>("src"))),
    handleBadStrips_(conf.getParameter<bool>("handleBadStrips")),
    maxBadStrips_(conf.getParameter<int>("maxBadStrips")) {
      
      produces<Phase2TrackerCluster1DCollectionNew>();

      badStripToken_ = esConsumes<SiStripBadStrip, SiPhase2OuterTrackerBadStripRcd>();

}

void Phase2TrackerClusterMerger::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // Get the Clusters
  edm::Handle<Phase2TrackerCluster1DCollectionNew> inputClusters;
  event.getByToken(token_, inputClusters);

  // Global container for the merged clusters of each module
  auto outputClusters = std::make_unique<Phase2TrackerCluster1DCollectionNew>();

  const SiStripBadStrip* badStripPayload_ = &eventSetup.getData(badStripToken_);

  // Merging never reaches across modules, so each DetSet is handled on its own. No
  // geometry is needed: adjacency is decided from the packed digi channel alone.
  for (const auto& DSViter : *inputClusters) {
    Phase2TrackerCluster1DCollectionNew::FastFiller clusters(*outputClusters, DSViter.detId());
    Phase2TrackerClusterMergerAlgorithm algo;
    algo.mergeDetUnit(DSViter, clusters,
                      handleBadStrips_, badStripPayload_, maxBadStrips_);
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
  desc.add<bool>("handleBadStrips", true);
  desc.add<int>("maxBadStrips", 1);  
  descriptions.add("default_phase2TrackerClusterMerger", desc);
}

DEFINE_FWK_MODULE(Phase2TrackerClusterMerger);
