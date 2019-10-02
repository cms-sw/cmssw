#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderEDProducer.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

VectorHitBuilderEDProducer::VectorHitBuilderEDProducer(edm::ParameterSet const& conf)
    : offlinestubsTag_(conf.getParameter<std::string>("offlinestubs")),
      maxOfflinestubs_(conf.getParameter<int>("maxVectorHits")),
      algoTag_(conf.getParameter<std::string>("Algorithm")),
      //_clusterProducer(conf.getParameter<edm::InputTag>("Clusters")),
      readytobuild_(false) {
  clusterProducer_ =
      consumes<edmNew::DetSetVector<Phase2TrackerCluster1D>>(edm::InputTag(conf.getParameter<std::string>("Clusters")));

  produces<edmNew::DetSetVector<Phase2TrackerCluster1D>>("ClustersAccepted");
  produces<edmNew::DetSetVector<Phase2TrackerCluster1D>>("ClustersRejected");
  produces<VectorHitCollectionNew>(offlinestubsTag_ + "Accepted");
  produces<VectorHitCollectionNew>(offlinestubsTag_ + "Rejected");
  setupAlgorithm(conf);
}

VectorHitBuilderEDProducer::~VectorHitBuilderEDProducer() { delete stubsBuilder_; }

void VectorHitBuilderEDProducer::produce(edm::Event& event, const edm::EventSetup& es) {
  LogDebug("VectorHitBuilderEDProducer") << "VectorHitBuilderEDProducer::produce() begin";

  // get input clusters data
  edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clustersHandle;
  event.getByToken(clusterProducer_, clustersHandle);

  // create the final output collection
  std::unique_ptr<edmNew::DetSetVector<Phase2TrackerCluster1D>> outputClustersAccepted(
      new edmNew::DetSetVector<Phase2TrackerCluster1D>);
  std::unique_ptr<edmNew::DetSetVector<Phase2TrackerCluster1D>> outputClustersRejected(
      new edmNew::DetSetVector<Phase2TrackerCluster1D>);
  std::unique_ptr<VectorHitCollectionNew> outputVHAccepted(new VectorHitCollectionNew());
  std::unique_ptr<VectorHitCollectionNew> outputVHRejected(new VectorHitCollectionNew());

  if (readytobuild_)
    stubsBuilder_->initialize(es);
  else
    edm::LogError("VectorHitBuilderEDProducer") << "Impossible initialization of builder!!";

  // check on the input clusters
  stubsBuilder_->printClusters(*clustersHandle);

  // running the stub building algorithm
  //ERICA::output should be moved in the different algo classes?
  run(clustersHandle, *outputClustersAccepted, *outputClustersRejected, *outputVHAccepted, *outputVHRejected);

  unsigned int numberOfVectorHits = 0;
  edmNew::DetSetVector<VectorHit>::const_iterator DSViter;
  for (DSViter = (*outputVHAccepted).begin(); DSViter != (*outputVHAccepted).end(); DSViter++) {
    edmNew::DetSet<VectorHit>::const_iterator vh;
    for (vh = DSViter->begin(); vh != DSViter->end(); ++vh) {
      numberOfVectorHits++;
      LogDebug("VectorHitBuilderEDProducer") << "\t vectorhit in output " << *vh << std::endl;
    }
  }
  /*
  if(numberOfVectorHits > _maxOfflinestubs) {
    edm::LogError("VectorHitBuilderEDProducer") <<  "Limit on the number of stubs exceeded. An empty output collection will be produced instead.\n";
    VectorHitCollectionNew empty;
    empty.swap(outputAcc);
  }
*/
  // write output to file
  event.put(std::move(outputClustersAccepted), "ClustersAccepted");
  event.put(std::move(outputClustersRejected), "ClustersRejected");
  event.put(std::move(outputVHAccepted), offlinestubsTag_ + "Accepted");
  event.put(std::move(outputVHRejected), offlinestubsTag_ + "Rejected");

  //  LogDebug("VectorHitBuilderEDProducer") << " Executing " << _algoTag << " resulted in " << numberOfVectorHits << ".";
  LogDebug("VectorHitBuilderEDProducer") << "found\n" << numberOfVectorHits << " .\n";
}

void VectorHitBuilderEDProducer::setupAlgorithm(edm::ParameterSet const& conf) {
  if (algoTag_ == "VectorHitBuilderAlgorithm") {
    stubsBuilder_ = new VectorHitBuilderAlgorithm(conf);
    readytobuild_ = true;
  } else {
    edm::LogError("VectorHitBuilderEDProducer") << " Choice " << algoTag_ << " is invalid.\n";
    readytobuild_ = false;
  }
}

void VectorHitBuilderEDProducer::run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
                                     VectorHitCollectionNew& outputAcc,
                                     VectorHitCollectionNew& outputRej) {
  if (!readytobuild_) {
    edm::LogError("VectorHitBuilderEDProducer") << " No stub builder algorithm was found - cannot run!";
    return;
  }

  stubsBuilder_->run(clusters, outputAcc, outputRej, clustersAcc, clustersRej);
}
void VectorHitBuilderEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("offlinestubs", "vectorHits");
  desc.add<int>("maxVectorHits", 999999999);
  desc.add<std::string>("Algorithm", "VectorHitBuilderAlgorithm");
  desc.add<edm::ESInputTag>("CPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"));
  desc.add<std::vector<double>>("BarrelCut",
                                {
                                    0.0,
                                    0.05,
                                    0.06,
                                    0.08,
                                    0.09,
                                    0.12,
                                    0.2,
                                });
  desc.add<std::string>("Clusters", "siPhase2Clusters");
  desc.add<int>("maxVectorHitsInAStack", 999);
  desc.add<std::vector<double>>("EndcapCut",
                                {
                                    0.0,
                                    0.1,
                                    0.1,
                                    0.1,
                                    0.1,
                                    0.1,
                                });
  descriptions.add("siPhase2VectorHits", desc);
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(VectorHitBuilderEDProducer);
