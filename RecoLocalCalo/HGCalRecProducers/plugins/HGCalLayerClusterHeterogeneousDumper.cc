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
      : token_{consumes(iConfig.getParameter<edm::InputTag>("src"))} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("hltHgcalSoARecHitsLayerClustersProducer"));
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID iStream, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override {
    auto const& data = iEvent.get(token_);

    auto const view = data.view();
    std::cout << fmt::format("view.numberOfClusters() = {}", view.numberOfClustersScalar()) << std::endl;
    for (int i = 0; i < data->metadata().size(); ++i) {
      std::cout << fmt::format("view[{}].clusterIndex() = {}", i, view.clusterIndex(i)) << std::endl;
    }
  }

private:
  edm::EDGetTokenT<HGCalSoARecHitsExtraHostCollection> const token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterHeterogeneousDumper);
