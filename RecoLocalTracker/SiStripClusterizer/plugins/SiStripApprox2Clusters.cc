#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include <vector>
#include <memory>

class SiStripApprox2Clusters : public edm::stream::EDProducer<> {
public:
  explicit SiStripApprox2Clusters(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag inputApproxClusters;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripApproximateCluster>> clusterToken;
};

SiStripApprox2Clusters::SiStripApprox2Clusters(const edm::ParameterSet& conf) {
  inputApproxClusters = conf.getParameter<edm::InputTag>("inputApproxClusters");
  clusterToken = consumes<edmNew::DetSetVector<SiStripApproximateCluster>>(inputApproxClusters);
  produces<edmNew::DetSetVector<SiStripCluster>>();
}

void SiStripApprox2Clusters::produce(edm::Event& event, edm::EventSetup const&) {
  auto result = std::make_unique<edmNew::DetSetVector<SiStripCluster>>();
  const auto& clusterCollection = event.get(clusterToken);

  for (const auto& detClusters : clusterCollection) {
    edmNew::DetSetVector<SiStripCluster>::FastFiller ff{*result, detClusters.id()};

    for (const auto& cluster : detClusters) {
      ff.push_back(SiStripCluster( cluster ));
    }
  }

  event.put(std::move(result));
}

void SiStripApprox2Clusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputApproxClusters", edm::InputTag("siStripClusters"));

  descriptions.add("SiStripApprox2Clusters", desc);
}

DEFINE_FWK_MODULE(SiStripApprox2Clusters);
