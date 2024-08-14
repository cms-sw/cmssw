#include <iostream>
#include <vector>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"

#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersDeviceCollection.h"
#include "DataFormats/HGCalReco/interface/alpaka/HGCalSoAClustersFilteredMaskDeviceCollection.h"


namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackstersSoAProducer : public stream::EDProducer<> {
    public:
      TrackstersSoAProducer(edm::ParameterSet const& config)
      : deviceTokenSoAClusters_(consumes(config.getParameter<edm::InputTag>("layer_clusters"))),
        cluster_mask_token_(consumes(config.getParameter<edm::InputTag>("clusters_mask_soa")))
      {}

      ~TrackstersSoAProducer() override = default;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        // hgcalMultiClusters
        edm::ParameterSetDescription desc;
        desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hltHgcalSoALayerClustersProducer"));
        desc.add<edm::InputTag>("clusters_mask_soa", edm::InputTag("hltFilteredLayerClustersSoAProducer"));
        descriptions.addWithDefaultLabel(desc);
      }

      void produce(device::Event& evt, device::EventSetup const& es) override {
        std::cout << " TrackstersSoAProducer\n";
      }

    private:

      device::EDGetToken<HGCalSoAClustersDeviceCollection> const deviceTokenSoAClusters_;
      device::EDGetToken<HGCalSoAClustersFilteredMaskDeviceCollection> const cluster_mask_token_;
      
  };

}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackstersSoAProducer);