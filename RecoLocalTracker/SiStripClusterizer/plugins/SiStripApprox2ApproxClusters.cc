

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

class SiStripApprox2ApproxClusters : public edm::stream::EDProducer<> {
public:
  explicit SiStripApprox2ApproxClusters(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::InputTag inputApproxClusters;
  uint8_t approxVersion;
  std::string approxVersionS;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripApproximateCluster>> clusterToken;
};

SiStripApprox2ApproxClusters::SiStripApprox2ApproxClusters(const edm::ParameterSet& conf) {
  inputApproxClusters = conf.getParameter<edm::InputTag>("inputApproxClusters");
  approxVersionS = conf.getParameter<std::string>("approxVersion");

  approxVersion = -1;

  if (approxVersionS == "ORIGINAL")
    approxVersion = 0;
  else if (approxVersionS == "FULL_WIDTH")
    approxVersion = 1;
  else if (approxVersionS == "BARY_RES_0.1")
    approxVersion = 2;
  else if (approxVersionS == "BARY_CHARGE_RES_0.1")
    approxVersion = 3;

  clusterToken = consumes<edmNew::DetSetVector<SiStripApproximateCluster>>(inputApproxClusters);
  produces<edmNew::DetSetVector<SiStripApproximateCluster>>();
}

void SiStripApprox2ApproxClusters::produce(edm::Event& event, edm::EventSetup const&) {
  auto result = std::make_unique<edmNew::DetSetVector<SiStripApproximateCluster>>();
  const auto& clusterCollection = event.get(clusterToken);

  for (const auto& detClusters : clusterCollection) {
    edmNew::DetSetVector<SiStripApproximateCluster>::FastFiller ff{*result, detClusters.id()};

    for (const auto& cluster : detClusters) {
      float barycenter = cluster.barycenter();
      uint8_t width = cluster.width();
      float avgCharge = cluster.avgCharge();
      bool filter = cluster.filter();
      bool isSaturated = cluster.isSaturated();

      switch (approxVersion) {
        case 0:  //ORIGINAL
          barycenter = std::round(barycenter);
          if (width > 0x3F)
            width = 0x3F;
          avgCharge = std::round(avgCharge);
          break;
        case 1:  //FULL_WIDTH
          barycenter = std::round(barycenter);
          avgCharge = std::round(avgCharge);
          break;
        case 2:  //BARY_RES_0.1
          barycenter = std::round(barycenter * 10) / 10;
          if (width > 0x3F)
            width = 0x3F;
          avgCharge = std::round(avgCharge);
          break;
        case 3:  //BARY_CHARGE_RES_0.1
          barycenter = std::round(barycenter * 10) / 10;
          if (width > 0x3F)
            width = 0x3F;
          avgCharge = std::round(avgCharge * 10) / 10;
          break;
      }

      ff.push_back(SiStripApproximateCluster(barycenter, width, avgCharge, filter, isSaturated));
    }
  }

  event.put(std::move(result));
}

void SiStripApprox2ApproxClusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputApproxClusters", edm::InputTag("siStripClusters"));
  desc.add<std::string>("approxVersion", std::string("ORIGINAL"));

  descriptions.add("SiStripApprox2ApproxClusters", desc);
}

DEFINE_FWK_MODULE(SiStripApprox2ApproxClusters);
