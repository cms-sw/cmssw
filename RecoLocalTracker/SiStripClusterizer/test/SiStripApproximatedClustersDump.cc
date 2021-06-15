#include "RecoLocalTracker/SiStripClusterizer/test/SiStripApproximatedClustersDump.h"

SiStripApproximatedClustersDump::SiStripApproximatedClustersDump(const edm::ParameterSet& conf) {
  inputTagClusters = conf.getParameter<edm::InputTag>("approximatedClustersTag");
  clusterToken = consumes<edmNew::DetSetVector<SiStripApproximateCluster>>(inputTagClusters);

  usesResource("TFileService");

  outNtuple = fs->make<TTree>("ApproxClusters", "ApproxClusters");
  outNtuple->Branch("event", &eventN, "event/i");
  outNtuple->Branch("detId", &detId, "detId/i");
  outNtuple->Branch("barycenter", &barycenter, "barycenter/F");
  outNtuple->Branch("width", &width, "width/b");
  outNtuple->Branch("charge", &avCharge, "charge/b");
}

SiStripApproximatedClustersDump::~SiStripApproximatedClustersDump() {}

void SiStripApproximatedClustersDump::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<edmNew::DetSetVector<SiStripApproximateCluster>> clusterCollection = event.getHandle(clusterToken);

  for (const auto& detClusters : *clusterCollection) {
    detId = detClusters.detId();
    eventN = event.id().event();

    for (const auto& cluster : detClusters) {
      barycenter = cluster.barycenter();
      width = cluster.width();
      avCharge = cluster.avgCharge();
      outNtuple->Fill();
    }
  }
}

void SiStripApproximatedClustersDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("approximatedClustersTag", edm::InputTag("SiStripClusters2ApproxClusters"));
  descriptions.add("SiStripApproximatedClustersDump", desc);
}
