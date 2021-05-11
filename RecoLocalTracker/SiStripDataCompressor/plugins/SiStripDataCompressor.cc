#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripDetSetCompressedCluster/interface/SiStripDetSetCompressedCluster.h"

#include "RecoLocalTracker/SiStripDataCompressor/interface/SiStripCompressionAlgorithm.h"

#include <vector>
#include <memory>

class SiStripDataCompressor : public edm::stream::EDProducer<> {
public:
  explicit SiStripDataCompressor(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<SiStripCompressionAlgorithm> algorithm;

  edm::InputTag inputTagClusters;
  edm::EDGetTokenT<vclusters_t> clusterToken;
};

SiStripDataCompressor::SiStripDataCompressor(const edm::ParameterSet& conf) {
  inputTagClusters = conf.getParameter<edm::InputTag>("clustersToBeCompressed");
  clusterToken = consumes<edmNew::DetSetVector<SiStripCluster> >(inputTagClusters);

  produces<vcomp_clusters_t>();
}

void SiStripDataCompressor::produce(edm::Event& event, const edm::EventSetup& es) {
  auto outClusters = std::make_unique<vcomp_clusters_t>();

  edm::Handle<vclusters_t> inClusters;
  event.getByToken(clusterToken, inClusters);

  algorithm->compress(*inClusters, *outClusters);

  event.put(std::move(outClusters));
}

void SiStripDataCompressor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("clustersToBeCompressed", edm::InputTag("siStripClusters"));

  descriptions.add("SiStripDataCompressor", desc);
}

DEFINE_FWK_MODULE(SiStripDataCompressor);
