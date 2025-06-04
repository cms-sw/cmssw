#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "CondFormats/DataRecord/interface/HcalPFCutsRcd.h"
#include "CondTools/Hcal/interface/HcalPFCutsHandler.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/SynchronizingEDProducer.h"

#include "RecoParticleFlow/PFClusterProducer/plugins/alpaka/PFMultiDepthClusterizer_Alpaka.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsDeviceCollection.h"
#include "RecoParticleFlow/PFClusterProducer/interface/alpaka/PFMultiDepthClusteringVarsHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/alpaka/PFRecHitDeviceCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PFMultiDepthClusterSoAProducer : public stream::SynchronizingEDProducer<> {
  public:
    PFMultiDepthClusterSoAProducer(edm::ParameterSet const& config);

    void acquire(device::Event const& event, device::EventSetup const&) override;
    void produce(device::Event& event, device::EventSetup const&) override;
    //
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    //
    const device::EDGetToken<reco::PFClusterDeviceCollection> inputPFClusterSoA_Token_;
    const device::EDGetToken<reco::PFRecHitFractionDeviceCollection> inputPFRecHitFractionSoA_Token_;
    const device::EDGetToken<reco::PFRecHitDeviceCollection> inputPFRecHitSoA_Token_;

    const edm::EDGetTokenT<cms_uint32_t> inputPFClustersNum_Token_;
    const edm::EDGetTokenT<cms_uint32_t> inputPFRecHitNum_Token_;

    const device::EDPutToken<reco::PFClusterDeviceCollection> outputPFClusterSoA_Token_;
    const device::EDPutToken<reco::PFRecHitFractionDeviceCollection> outputPFRHFractionsSoA_Token_;
    //
    cms::alpakatools::host_buffer<uint32_t> nClusters_;
    //
    const bool synchronise_;
    //
    edm::ParameterSet conf_;
    //
    std::optional<PFMultiDepthClusterizer_Alpaka> clusterizer_;
  };

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
  DEFINE_FWK_ALPAKA_MODULE(PFMultiDepthClusterSoAProducer);

  void PFMultiDepthClusterSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("clustersSource", {});
    desc.add<edm::ParameterSetDescription>("energyCorrector", {});
    {
      edm::ParameterSetDescription pset0;
      pset0.add<double>("minFractionToKeep", 1e-07);
      pset0.add<double>("nSigmaEta", 2.0);
      pset0.add<double>("nSigmaPhi", 2.0);
      desc.add<edm::ParameterSetDescription>("pfClusterBuilder", pset0);
    }
    descriptions.addWithDefaultLabel(desc);
  }

  PFMultiDepthClusterSoAProducer::PFMultiDepthClusterSoAProducer(const edm::ParameterSet& config)
      : SynchronizingEDProducer(config),
        inputPFClusterSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfClusters"))},
        inputPFRecHitFractionSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
        inputPFRecHitSoA_Token_{consumes(config.getParameter<edm::InputTag>("pfRecHits"))},
        outputPFClusterSoA_Token_{produces()},
        outputPFRHFractionsSoA_Token_{produces()},
        nClusters_{cms::alpakatools::make_host_buffer<uint32_t, Platform>()},
        synchronise_(config.getParameter<bool>("synchronise")),
        conf_(config) {}

  void PFMultiDepthClusterSoAProducer::acquire(device::Event const& event, device::EventSetup const&) {
    *nClusters_ = event.get(inputPFClustersNum_Token_);

    if (!clusterizer_) {
      // Initialize clusterizer at first event
      clusterizer_.emplace(event.queue(), conf_.getParameterSet("pfClusterBuilder"), *nClusters_);
    }
  }

  void PFMultiDepthClusterSoAProducer::produce(device::Event& event, const device::EventSetup& eventSetup) {
    const reco::PFClusterDeviceCollection& pfClusters = event.get(inputPFClusterSoA_Token_);
    const reco::PFRecHitDeviceCollection& pfRecHits = event.get(inputPFRecHitSoA_Token_);

    const reco::PFRecHitFractionDeviceCollection& pfRecHitFractions = event.get(inputPFRecHitFractionSoA_Token_);

    std::optional<reco::PFRecHitFractionDeviceCollection> outPFRHFractions;
    std::optional<reco::PFClusterDeviceCollection> outPFClusters;

    if (*nClusters_ > 0) {
      int nRH_ = event.get(inputPFRecHitNum_Token_);
      //
      outPFClusters.emplace(*nClusters_, event.queue());
      outPFRHFractions.emplace(nRH_, event.queue());
      //
      clusterizer_->apply(event.queue(), *outPFClusters, *outPFRHFractions, pfClusters, pfRecHitFractions, pfRecHits);
    } else {
      outPFClusters.emplace(0, event.queue());
      outPFRHFractions.emplace(0, event.queue());
    }

    if (synchronise_)
      alpaka::wait(event.queue());

    event.emplace(outputPFClusterSoA_Token_, std::move(*outPFClusters));
    event.emplace(outputPFRHFractionsSoA_Token_, std::move(*outPFRHFractions));
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
