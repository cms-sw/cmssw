#include <Eigen/Core>
#include <Eigen/Dense>
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusterParamsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFClusterSoAProducerKernel.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class PFClusterSoAProducer : public stream::EDProducer<> {
  public:
    PFClusterSoAProducer(edm::ParameterSet const& config)
        : pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
          topologyToken_(esConsumes(config.getParameter<edm::ESInputTag>("topology"))),
          inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
          outputPFClusterSoA_Token_{produces()},
          outputPFRHFractionSoA_Token_{produces()},
          synchronise_(config.getParameter<bool>("synchronise")),
          pfRecHitFractionAllocation_(config.getParameter<int>("pfRecHitFractionAllocation")) {}

    void produce(device::Event& event, device::EventSetup const& setup) override {
      const reco::PFClusterParamsDeviceCollection& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHCALTopologyDeviceCollection& topology = setup.getData(topologyToken_);
      const reco::PFRecHitHostCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);
      const int nRH = pfRecHits->size();

      reco::PFClusteringVarsDeviceCollection pfClusteringVars{nRH, event.queue()};
      reco::PFClusteringEdgeVarsDeviceCollection pfClusteringEdgeVars{(nRH * 8), event.queue()};
      reco::PFClusterDeviceCollection pfClusters{nRH, event.queue()};
      reco::PFRecHitFractionDeviceCollection pfrhFractions{nRH * pfRecHitFractionAllocation_, event.queue()};

      PFClusterProducerKernel kernel(event.queue(), pfRecHits);
      kernel.execute(
          event.queue(), params, topology, pfClusteringVars, pfClusteringEdgeVars, pfRecHits, pfClusters, pfrhFractions);

      if (synchronise_)
        alpaka::wait(event.queue());

      event.emplace(outputPFClusterSoA_Token_, std::move(pfClusters));
      event.emplace(outputPFRHFractionSoA_Token_, std::move(pfrhFractions));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pfRecHits");
      desc.add<edm::ESInputTag>("pfClusterParams");
      desc.add<edm::ESInputTag>("topology");
      desc.add<bool>("synchronise");
      desc.add<int>("pfRecHitFractionAllocation", 120);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<reco::PFClusterParamsDeviceCollection, JobConfigurationGPURecord> pfClusParamsToken;
    const device::ESGetToken<reco::PFRecHitHCALTopologyDeviceCollection, PFRecHitHCALTopologyRecord> topologyToken_;
    const edm::EDGetTokenT<reco::PFRecHitHostCollection> inputPFRecHitSoA_Token_;
    const device::EDPutToken<reco::PFClusterDeviceCollection> outputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> outputPFRHFractionSoA_Token_;
    const bool synchronise_;
    const int pfRecHitFractionAllocation_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFClusterSoAProducer);
