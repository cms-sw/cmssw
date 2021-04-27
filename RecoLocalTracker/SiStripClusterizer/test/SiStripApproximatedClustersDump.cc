#include "RecoLocalTracker/SiStripClusterizer/test/SiStripApproximatedClustersDump.h"


SiStripApproximatedClustersDump::SiStripApproximatedClustersDump(const edm::ParameterSet& conf) {
  inputTagClusters = conf.getParameter< edm::InputTag >("approximatedClustersTag");
  clusterToken = consumes< edmNew::DetSetVector<SiStripApproximateClusterv1>>(inputTagClusters);
  
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
  
  edm::Handle<edmNew::DetSetVector<SiStripApproximateClusterv1>> inClusters;
  event.getByToken(clusterToken, inClusters);

  
  for (edmNew::DetSetVector<SiStripApproximateClusterv1>::const_iterator itApprox = inClusters->begin(); itApprox!= inClusters->end(); itApprox++) {
    detId = itApprox->detId();
    eventN = event.id().event();
    
    for (edmNew::DetSet<SiStripApproximateClusterv1>::const_iterator itClusters = itApprox->begin(); itClusters!= itApprox->end(); itClusters++){

      barycenter = itClusters->barycenter();
      width = itClusters->width();
      avCharge=itClusters->avgCharge();
      outNtuple->Fill();
      
    }
  }
}


