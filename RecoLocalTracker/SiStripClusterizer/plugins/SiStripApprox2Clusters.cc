#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
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

class SiStripApprox2Clusters : public edm::global::EDProducer<> {
public:
  explicit SiStripApprox2Clusters(const edm::ParameterSet& conf);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripApproximateCluster>> clusterToken_;
};

SiStripApprox2Clusters::SiStripApprox2Clusters(const edm::ParameterSet& conf) {
  clusterToken_ = consumes<edmNew::DetSetVector<SiStripApproximateCluster>>(
      conf.getParameter<edm::InputTag>("inputApproxClusters"));
  produces<edmNew::DetSetVector<SiStripCluster>>();
}

void SiStripApprox2Clusters::produce(edm::StreamID id, edm::Event& event, const edm::EventSetup& iSetup) const {
  auto result = std::make_unique<edmNew::DetSetVector<SiStripCluster>>();
  const auto& clusterCollection = event.get(clusterToken_);

  for (const auto& detClusters : clusterCollection) {
    edmNew::DetSetVector<SiStripCluster>::FastFiller ff{*result, detClusters.id()};

    for (const auto& cluster : detClusters) {
      ff.push_back(SiStripCluster(cluster));
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
