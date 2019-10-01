#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderEDProducer.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

VectorHitBuilderEDProducer::VectorHitBuilderEDProducer(edm::ParameterSet const& conf)
    : offlinestubsTag(conf.getParameter<std::string>("offlinestubs")),
      maxOfflinestubs(conf.getParameter<int>("maxVectorHits")),
      algoTag(conf.getParameter<std::string>("Algorithm")),
      //clusterProducer(conf.getParameter<edm::InputTag>("Clusters")),
      readytobuild(false) {
  clusterProducer = consumes<edmNew::DetSetVector<Phase2TrackerCluster1D> >(
      edm::InputTag(conf.getParameter<std::string>("Clusters")));

  produces<edmNew::DetSetVector<Phase2TrackerCluster1D> >("ClustersAccepted");
  produces<edmNew::DetSetVector<Phase2TrackerCluster1D> >("ClustersRejected");
  produces<VectorHitCollectionNew>(offlinestubsTag + "Accepted");
  produces<VectorHitCollectionNew>(offlinestubsTag + "Rejected");
  setupAlgorithm(conf);
}

VectorHitBuilderEDProducer::~VectorHitBuilderEDProducer() { delete stubsBuilder; }

void VectorHitBuilderEDProducer::produce(edm::Event& event, const edm::EventSetup& es) {
  LogDebug("VectorHitBuilderEDProducer") << "VectorHitBuilderEDProducer::produce() begin";

  // get input clusters data
  edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D> > clustersHandle;
  event.getByToken(clusterProducer, clustersHandle);

  // create the final output collection
  std::unique_ptr<edmNew::DetSetVector<Phase2TrackerCluster1D> > outputClustersAccepted(
      new edmNew::DetSetVector<Phase2TrackerCluster1D>);
  std::unique_ptr<edmNew::DetSetVector<Phase2TrackerCluster1D> > outputClustersRejected(
      new edmNew::DetSetVector<Phase2TrackerCluster1D>);
  std::unique_ptr<VectorHitCollectionNew> outputVHAccepted(new VectorHitCollectionNew());
  std::unique_ptr<VectorHitCollectionNew> outputVHRejected(new VectorHitCollectionNew());

  if (readytobuild)
    stubsBuilder->initialize(es);
  else
    edm::LogError("VectorHitBuilderEDProducer") << "Impossible initialization of builder!!";

  // check on the input clusters
  stubsBuilder->printClusters(*clustersHandle);

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
  if(numberOfVectorHits > maxOfflinestubs) {
    edm::LogError("VectorHitBuilderEDProducer") <<  "Limit on the number of stubs exceeded. An empty output collection will be produced instead.\n";
    VectorHitCollectionNew empty;
    empty.swap(outputAcc);
  }
*/
  // write output to file
  event.put(std::move(outputClustersAccepted), "ClustersAccepted");
  event.put(std::move(outputClustersRejected), "ClustersRejected");
  event.put(std::move(outputVHAccepted), offlinestubsTag + "Accepted");
  event.put(std::move(outputVHRejected), offlinestubsTag + "Rejected");

  //  LogDebug("VectorHitBuilderEDProducer") << " Executing " << algoTag << " resulted in " << numberOfVectorHits << ".";
  LogDebug("VectorHitBuilderEDProducer") << "found\n" << numberOfVectorHits << " .\n";
}

void VectorHitBuilderEDProducer::setupAlgorithm(edm::ParameterSet const& conf) {
  if (algoTag == "VectorHitBuilderAlgorithm") {
    stubsBuilder = new VectorHitBuilderAlgorithm(conf);
    readytobuild = true;
  } else {
    edm::LogError("VectorHitBuilderEDProducer") << " Choice " << algoTag << " is invalid.\n";
    readytobuild = false;
  }
}

void VectorHitBuilderEDProducer::run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D> > clusters,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
                                     VectorHitCollectionNew& outputAcc,
                                     VectorHitCollectionNew& outputRej) {
  if (!readytobuild) {
    edm::LogError("VectorHitBuilderEDProducer") << " No stub builder algorithm was found - cannot run!";
    return;
  }

  stubsBuilder->run(clusters, outputAcc, outputRej, clustersAcc, clustersRej);
}

#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(VectorHitBuilderEDProducer);
