#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithmBase.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "RecoLocalTracker/SiPhase2VectorHitBuilder/interface/VectorHitBuilderAlgorithm.h"
#include "DataFormats/TrackerRecHit2D/interface/VectorHit.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

class VectorHitBuilderEDProducer : public edm::stream::EDProducer<> {
public:
  explicit VectorHitBuilderEDProducer(const edm::ParameterSet&);
  ~VectorHitBuilderEDProducer() override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
           VectorHitCollectionNew& outputAcc,
           VectorHitCollectionNew& outputRej);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  const VectorHitBuilderAlgorithm* algo() const { return stubsBuilder_; };

private:
  const VectorHitBuilderAlgorithm* stubsBuilder_;
  std::string offlinestubsTag_;
  unsigned int maxOfflinestubs_;
  edm::ESGetToken<VectorHitBuilderAlgorithm, TkPhase2OTCPERecord> stubsBuilderToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusterProducer_;
};

VectorHitBuilderEDProducer::~VectorHitBuilderEDProducer() {}

VectorHitBuilderEDProducer::VectorHitBuilderEDProducer(edm::ParameterSet const& conf)
    : offlinestubsTag_(conf.getParameter<std::string>("offlinestubs")),
      maxOfflinestubs_(conf.getParameter<int>("maxVectorHits")),
      stubsBuilderToken_(esConsumes(conf.getParameter<edm::ESInputTag>("Algorithm"))) {
  clusterProducer_ =
      consumes<edmNew::DetSetVector<Phase2TrackerCluster1D>>(edm::InputTag(conf.getParameter<std::string>("Clusters")));

  produces<edmNew::DetSetVector<Phase2TrackerCluster1D>>("ClustersAccepted");
  produces<edmNew::DetSetVector<Phase2TrackerCluster1D>>("ClustersRejected");
  produces<VectorHitCollectionNew>(offlinestubsTag_ + "Accepted");
  produces<VectorHitCollectionNew>(offlinestubsTag_ + "Rejected");
}

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

  stubsBuilder_ = &es.getData(stubsBuilderToken_);
  // check on the input clusters
  stubsBuilder_->printClusters(*clustersHandle);

  // running the stub building algorithm
  //ERICA::output should be moved in the different algo classes?
  run(clustersHandle, *outputClustersAccepted, *outputClustersRejected, *outputVHAccepted, *outputVHRejected);

  unsigned int numberOfVectorHits = 0;
  for (const auto& DSViter : *outputVHAccepted) {
    for (const auto& vh : DSViter) {
      numberOfVectorHits++;
      LogDebug("VectorHitBuilderEDProducer") << "\t vectorhit in output " << vh << std::endl;
    }
  }
  // write output to file
  event.put(std::move(outputClustersAccepted), "ClustersAccepted");
  event.put(std::move(outputClustersRejected), "ClustersRejected");
  event.put(std::move(outputVHAccepted), offlinestubsTag_ + "Accepted");
  event.put(std::move(outputVHRejected), offlinestubsTag_ + "Rejected");

  LogDebug("VectorHitBuilderEDProducer") << "found\n" << numberOfVectorHits << " .\n";
}

void VectorHitBuilderEDProducer::run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
                                     VectorHitCollectionNew& outputAcc,
                                     VectorHitCollectionNew& outputRej) {
  stubsBuilder_->run(clusters, outputAcc, outputRej, clustersAcc, clustersRej);
}
void VectorHitBuilderEDProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("offlinestubs", "vectorHits");
  desc.add<int>("maxVectorHits", 999999999);
  desc.add<edm::ESInputTag>("Algorithm", edm::ESInputTag("", "SiPhase2VectorHitMatcher"));
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
DEFINE_FWK_MODULE(VectorHitBuilderEDProducer);
#include "FWCore/Utilities/interface/typelookup.h"
TYPELOOKUP_DATA_REG(VectorHitBuilderEDProducer);
