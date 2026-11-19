#include "DataFormats/CaloRecHit/interface/CaloClusterHostCollection.h"
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

#include <iostream>
#include <limits>

#include <fmt/format.h>

class HGCalLayerClusterHeterogeneousSoADumper : public edm::global::EDAnalyzer<> {
public:
  HGCalLayerClusterHeterogeneousSoADumper(edm::ParameterSet const& iConfig)
      : token_{consumes(iConfig.getParameter<edm::InputTag>("src"))} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("hltHgcalSoALayerClustersProducer"));
    descriptions.addWithDefaultLabel(desc);
  }

  void analyze(edm::StreamID iStream, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override {
    auto const& data = iEvent.get(token_);

    auto const view = data.const_view();
    const int size = view.position().metadata().size();
    std::cout << fmt::format("hgcalSoALayerClustersProducer size = {}", size) << std::endl;
    for (int i = 0; i < size; ++i) {
      std::cout << fmt::format("CLUSTERS_SOA {}, energy = {:.{}f}, x = {:.{}f}, y = {:.{}f}, z= {:.{}f}",
                               i,
                               view.energy()[i].energy(),
                               std::numeric_limits<float>::max_digits10,
                               view.position()[i].x(),
                               std::numeric_limits<float>::max_digits10,
                               view.position()[i].y(),
                               std::numeric_limits<float>::max_digits10,
                               view.position()[i].z(),
                               std::numeric_limits<float>::max_digits10)
                << std::endl;
    }
  }

private:
  edm::EDGetTokenT<reco::CaloClusterHostCollection> const token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterHeterogeneousSoADumper);
