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

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class VectorHitBuilderEDProducer : public edm::stream::EDProducer<> {
public:
  explicit VectorHitBuilderEDProducer(const edm::ParameterSet&);
  ~VectorHitBuilderEDProducer() override = default;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
           edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
           VectorHitCollection& outputAcc,
           VectorHitCollection& outputRej);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  const VectorHitBuilderAlgorithm* algo() const { return stubsBuilder_; }

private:
  const VectorHitBuilderAlgorithm* stubsBuilder_;
  std::string offlinestubsTag_;
  unsigned int maxOfflinestubs_;
  edm::ESGetToken<VectorHitBuilderAlgorithm, TkPhase2OTCPERecord> stubsBuilderToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusterProducer_;
};

VectorHitBuilderEDProducer::VectorHitBuilderEDProducer(edm::ParameterSet const& conf)
    : offlinestubsTag_(conf.getParameter<std::string>("offlinestubs")),
      maxOfflinestubs_(conf.getParameter<int>("maxVectorHits")),
      stubsBuilderToken_(esConsumes(conf.getParameter<edm::ESInputTag>("Algorithm"))) {
  clusterProducer_ =
      consumes<edmNew::DetSetVector<Phase2TrackerCluster1D>>(edm::InputTag(conf.getParameter<std::string>("Clusters")));

  produces<edmNew::DetSetVector<Phase2TrackerCluster1D>>("accepted");
  produces<edmNew::DetSetVector<Phase2TrackerCluster1D>>("rejected");
  produces<VectorHitCollection>("accepted");
  produces<VectorHitCollection>("rejected");
}

void VectorHitBuilderEDProducer::produce(edm::Event& event, const edm::EventSetup& es) {
  LogDebug("VectorHitBuilderEDProducer") << "VectorHitBuilderEDProducer::produce() begin";

  // get input clusters data
  auto clustersHandle = event.getHandle(clusterProducer_);
  // create the final output collection
  auto outputClustersAccepted = std::make_unique<edmNew::DetSetVector<Phase2TrackerCluster1D>>();
  auto outputClustersRejected = std::make_unique<edmNew::DetSetVector<Phase2TrackerCluster1D>>();
  std::unique_ptr<VectorHitCollection> outputVHAccepted(new VectorHitCollection());
  std::unique_ptr<VectorHitCollection> outputVHRejected(new VectorHitCollection());

  stubsBuilder_ = &es.getData(stubsBuilderToken_);
#ifdef EDM_ML_DEBUG
  // check on the input clusters
  stubsBuilder_->printClusters(*clustersHandle);
#endif  //EDM_ML_DEBUG

  // running the stub building algorithm
  //ERICA::output should be moved in the different algo classes?
  run(clustersHandle, *outputClustersAccepted, *outputClustersRejected, *outputVHAccepted, *outputVHRejected);
#ifdef EDM_ML_DEBUG
  unsigned int numberOfVectorHits = 0;
  for (const auto& dSViter : *outputVHAccepted) {
    for (const auto& vh : dSViter) {
      numberOfVectorHits++;
      LogDebug("VectorHitBuilderEDProducer") << "\t vectorhit in output " << vh;
    }
  }
  LogDebug("VectorHitBuilderEDProducer") << "found\n" << numberOfVectorHits << " .\n";
#endif  //EDM_ML_DEBUG
  // write output to file
  event.put(std::move(outputClustersAccepted), "accepted");
  event.put(std::move(outputClustersRejected), "rejected");
  event.put(std::move(outputVHAccepted), "accepted");
  event.put(std::move(outputVHRejected), "rejected");
}

void VectorHitBuilderEDProducer::run(edm::Handle<edmNew::DetSetVector<Phase2TrackerCluster1D>> clusters,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersAcc,
                                     edmNew::DetSetVector<Phase2TrackerCluster1D>& clustersRej,
                                     VectorHitCollection& outputAcc,
                                     VectorHitCollection& outputRej) {
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
