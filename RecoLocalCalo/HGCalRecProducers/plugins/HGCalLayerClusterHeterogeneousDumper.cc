#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtraHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <fmt/format.h>

class HGCalLayerClusterHeterogeneousDumper : public edm::global::EDAnalyzer<> {
public:
  HGCalLayerClusterHeterogeneousDumper(edm::ParameterSet const& iConfig)
      : deviceToken_{consumes(iConfig.getParameter<edm::InputTag>("srcDevice"))} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("srcDevice", edm::InputTag("hgCalSoARecHitsLayerClustersProducer"));
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID iStream, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override {
    auto const& deviceData = iEvent.get(deviceToken_);

    auto const deviceView = deviceData.view();
    std::cout << fmt::format("view.numberOfClusters() = {}", deviceView.numberOfClustersScalar()) << std::endl;
    for (int i = 0; i < deviceData->metadata().size(); ++i) {
      std::cout << fmt::format("view[{}].clusterIndex() = {}", i, deviceView.clusterIndex(i)) << std::endl;
    }
  }

private:
  edm::EDGetTokenT<HGCalSoARecHitsExtraHostCollection> const deviceToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterHeterogeneousDumper);
