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

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackstersSoAProducer : public stream::EDProducer<> {
    public:
      TrackstersSoAProducer(edm::ParameterSet const& config) {}
      ~TrackstersSoAProducer() override = default;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
        // hgcalMultiClusters
        edm::ParameterSetDescription desc;
        descriptions.addWithDefaultLabel(desc);
      }

      void produce(device::Event& evt, device::EventSetup const& es) override {
        std::cout << " TrackstersSoAProducer\n";
      }

    private:

  };

}

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackstersSoAProducer);
