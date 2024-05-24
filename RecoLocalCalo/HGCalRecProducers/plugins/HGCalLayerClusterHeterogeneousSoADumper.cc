#include "DataFormats/HGCalReco/interface/HGCalSoAClustersHostCollection.h"
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

class HGCalLayerClusterHeterogeneousSoADumper : public edm::global::EDAnalyzer<> {
public:
  HGCalLayerClusterHeterogeneousSoADumper(edm::ParameterSet const& iConfig)
      : deviceToken_{consumes(iConfig.getParameter<edm::InputTag>("srcDevice"))} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("srcDevice", edm::InputTag("hgCalSoALayerClustersProducer"));
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID iStream, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override {
    auto const& deviceData = iEvent.get(deviceToken_);

    auto const deviceView = deviceData.view();
    std::cout << fmt::format("hgcalSoALayerClustersProducer size = {}", deviceView.metadata().size()) << std::endl;
    for (int i = 0; i < deviceData->metadata().size(); ++i) {
      std::cout << fmt::format("CLUSTERS_SOA {}, energy = {:.{}f}, x = {:.{}f}, y = {:.{}f}, z= {:.{}f}",
                               i,
                               deviceView.energy(i),
                               std::numeric_limits<float>::max_digits10,
                               deviceView.x(i),
                               std::numeric_limits<float>::max_digits10,
                               deviceView.y(i),
                               std::numeric_limits<float>::max_digits10,
                               deviceView.z(i),
                               std::numeric_limits<float>::max_digits10)
                << std::endl;
    }
  }

private:
  edm::EDGetTokenT<HGCalSoAClustersHostCollection> const deviceToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterHeterogeneousSoADumper);
