#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"
#include "HeterogeneousCore/CUDACore/interface/JobConfigurationGPURecord.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFClusterParamsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFClusterSoAProducerKernel.h"
#include "RecoParticleFlow/PFRecHitProducer/interface/PFRecHitTopologyRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFClusterSoAProducer : public stream::SynchronizingEDProducer<> {
  public:
    PFClusterSoAProducer(edm::ParameterSet const& config)
        : pfClusParamsToken(esConsumes(config.getParameter<edm::ESInputTag>("pfClusterParams"))),
          topologyToken_(esConsumes(config.getParameter<edm::ESInputTag>("topology"))),
          inputPFRecHitHostSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
          inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
          outputPFClusterSoA_Token_{produces()},
          outputPFRHFractionSoA_Token_{produces()},
          numRHF_{cms::alpakatools::make_host_buffer<uint32_t, Platform>()},
          synchronise_(config.getParameter<bool>("synchronise")) {}

    void acquire(device::Event const& event, device::EventSetup const& setup) override {
      const reco::PFClusterParamsDeviceCollection& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHCALTopologyDeviceCollection& topology = setup.getData(topologyToken_);
      const reco::PFRecHitDeviceCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);
      int nRH = 0;
      if (pfRecHits->metadata().size() != 0) {
        // FIXME this is a quick workaround to let the device code use the device collection,
        // while being able to access the actual number of pf rechits on the host side.
        // It should replaced with a better and more general implementation, and the use of the
        // host collection should be removed.
        reco::PFRecHitHostCollection const& pfRecHitsHost = event.get(inputPFRecHitHostSoA_Token_);
        nRH = pfRecHitsHost->size();
      }

      pfClusteringVars_.emplace(nRH, event.queue());
      pfClusteringEdgeVars_.emplace(nRH * 8, event.queue());
      pfClusters_.emplace(nRH, event.queue());

      *numRHF_ = 0;

      if (nRH != 0) {
        PFClusterProducerKernel kernel(event.queue());
        kernel.seedTopoAndContract(event.queue(),
                                   params,
                                   topology,
                                   *pfClusteringVars_,
                                   *pfClusteringEdgeVars_,
                                   pfRecHits,
                                   nRH,
                                   *pfClusters_,
                                   numRHF_.data());
      }
    }

    void produce(device::Event& event, device::EventSetup const& setup) override {
      const reco::PFClusterParamsDeviceCollection& params = setup.getData(pfClusParamsToken);
      const reco::PFRecHitHCALTopologyDeviceCollection& topology = setup.getData(topologyToken_);
      const reco::PFRecHitDeviceCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);
      int nRH = 0;

      std::optional<reco::PFRecHitFractionDeviceCollection> pfrhFractions;

      if (pfRecHits->metadata().size() != 0) {
        reco::PFRecHitHostCollection const& pfRecHitsHost = event.get(inputPFRecHitHostSoA_Token_);
        nRH = pfRecHitsHost->size();
      }

      if (nRH != 0) {
        pfrhFractions.emplace(*numRHF_.data(), event.queue());
        PFClusterProducerKernel kernel(event.queue());
        kernel.cluster(event.queue(),
                       params,
                       topology,
                       *pfClusteringVars_,
                       *pfClusteringEdgeVars_,
                       pfRecHits,
                       nRH,
                       *pfClusters_,
                       *pfrhFractions);
      } else {
        pfrhFractions.emplace(0, event.queue());
      }

      if (synchronise_)
        alpaka::wait(event.queue());

      event.emplace(outputPFClusterSoA_Token_, std::move(*pfClusters_));
      event.emplace(outputPFRHFractionSoA_Token_, std::move(*pfrhFractions));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("pfRecHits", edm::InputTag(""));
      desc.add<edm::ESInputTag>("pfClusterParams", edm::ESInputTag(""));
      desc.add<edm::ESInputTag>("topology", edm::ESInputTag(""));
      desc.add<bool>("synchronise", false);
      descriptions.addWithDefaultLabel(desc);
    }

  private:
    const device::ESGetToken<reco::PFClusterParamsDeviceCollection, JobConfigurationGPURecord> pfClusParamsToken;
    const device::ESGetToken<reco::PFRecHitHCALTopologyDeviceCollection, PFRecHitHCALTopologyRecord> topologyToken_;
    const edm::EDGetTokenT<reco::PFRecHitHostCollection> inputPFRecHitHostSoA_Token_;
    const device::EDGetToken<reco::PFRecHitDeviceCollection> inputPFRecHitSoA_Token_;
    const device::EDPutToken<reco::PFClusterDeviceCollection> outputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> outputPFRHFractionSoA_Token_;
    cms::alpakatools::host_buffer<uint32_t> numRHF_;
    std::optional<reco::PFClusteringVarsDeviceCollection> pfClusteringVars_;
    std::optional<reco::PFClusteringEdgeVarsDeviceCollection> pfClusteringEdgeVars_;
    std::optional<reco::PFClusterDeviceCollection> pfClusters_;
    const bool synchronise_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PFClusterSoAProducer);
